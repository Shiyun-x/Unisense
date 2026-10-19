"""
Modality-Aware MoE LoRA FFN for Unisense decoder layers.

Design
------
For each configured modality (e.g. 'depth'), every token that belongs to
that modality is hard-routed to that modality's LoRA expert pool.  Within
the pool a learned soft router distributes the token among
``num_experts_per_modal`` experts.  Tokens whose modality has no
configured expert pool pass through the frozen base MLP unchanged.

Process-global state (not threading.local)
------------------------------------------
Rather than threading ``modality_ids`` through every layer of Qwen3.5's
forward (which would require modifying the HuggingFace model internals),
we store it in a module-level dict shared across all threads in the process.
Before calling ``self.model(...)`` in ``UnisenseQwen35ForCausalLM.forward``,
call ``set_modality_ids(ids)``.  Each ``ModalityAwareMoEFFN`` reads the
dict in its forward.  We intentionally do NOT use ``threading.local``
because HuggingFace's ``use_reentrant=False`` gradient checkpointing
triggers recomputation from a background C++ thread that would not inherit
thread-local values, causing a CheckpointError.

Modality ID encoding
--------------------
  MODALITY_TEXT       = 0  (default / text tokens / padding)
  MODALITY_IMAGE      = 1  (RGB image tokens)
  MODALITY_SPEECH     = 2  (speech tokens)
  MODALITY_DEPTH      = 3  (depth image tokens)
  MODALITY_POINTCLOUD = 4  (3D point cloud tokens)
  MODALITY_IMU        = 5  (IMU motion sensor tokens)
  MODALITY_TACTILE    = 6  (tactile image tokens)
  MODALITY_THERMAL    = 7  (thermal image tokens)
  MODALITY_RADAR      = 8  (mmWave radar point-cloud tokens)
  MODALITY_SMELL      = 9  (multichannel gas sensor tokens)
  MODALITY_MOTION     = 11 (skeleton trajectory tokens)

Expert architecture (proj_mode)
--------------------------------
  "up_down"      : delta = B(A(x))  [standard LoRA, 2 matrices per expert]
  "gate_up_down" : delta = down(act(gate(x)) * up(x))  [SwiGLU-style, 3 matrices per expert]
                   reuses the same activation as the base Qwen3_5MLP

Layer selection (apply_moe_ffn_surgery)
----------------------------------------
  group_indices  : list of group indices (0-based) to patch.
                   For Qwen3.5-2B (pattern: lin,lin,lin,full × 6), there are 6 groups (0-5).
                   E.g. [0, 2, 5] patches groups 0, 2, and 5.
  full_attn_only : if True, only patch the full_attention layers within those groups;
                   if False, patch all layers in the selected groups.

Naming convention
-----------------
Expert parameters live under ``moe_experts`` / ``moe_routers`` (not ``lora_*``) so
``_ensure_trainable_params_in_graph`` in unisense_arch_qwen.py does NOT exclude them
(that helper skips names containing 'lora_').
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Modality ID constants
# ---------------------------------------------------------------------------

MODALITY_TEXT       = 0
MODALITY_IMAGE      = 1
MODALITY_SPEECH     = 2
MODALITY_DEPTH      = 3
MODALITY_POINTCLOUD = 4
MODALITY_IMU        = 5
MODALITY_TACTILE    = 6
MODALITY_THERMAL    = 7
MODALITY_RADAR      = 8
MODALITY_SMELL      = 9
MODALITY_MOTION     = 11

MODALITY_ID_MAP: dict = {
    'text':       MODALITY_TEXT,
    'image':      MODALITY_IMAGE,
    'speech':     MODALITY_SPEECH,
    'depth':      MODALITY_DEPTH,
    'pointcloud': MODALITY_POINTCLOUD,
    'imu':        MODALITY_IMU,
    'tactile':    MODALITY_TACTILE,
    'thermal':    MODALITY_THERMAL,
    'radar':      MODALITY_RADAR,
    'smell':      MODALITY_SMELL,
    'motion':     MODALITY_MOTION,
}

# ---------------------------------------------------------------------------
# Process-global state (NOT threading.local)
# ---------------------------------------------------------------------------
# We deliberately use a plain module-level dict instead of threading.local.
#
# Reason: HuggingFace gradient checkpointing uses use_reentrant=False, which
# triggers recomputation via unpack_hook called from a PyTorch C++ background
# thread.  threading.local() values set on the main Python thread are NOT
# visible on that background thread, causing ModalityAwareMoEFFN.forward() to
# see None during recomputation and take a different code path → CheckpointError
# (mismatched saved tensor counts).
#
# Safety: DDP training uses separate OS processes (not threads), so each
# process has its own module globals.  Within a process, autograd recomputation
# is single-step sequential, so no concurrent writes occur.

_global_moe_state: dict = {"modality_ids": None}

# ---------------------------------------------------------------------------
# Process-global aux-loss accumulators
# ---------------------------------------------------------------------------
# Each ModalityAwareMoEFFN.forward() appends its per-layer balance / diversity
# loss here during the *original* forward pass.  UnisenseQwen35ForCausalLM.forward()
# clears these lists before self.model(...) and sums them afterwards.
#
# Gradient-checkpointing note: with use_reentrant=False, checkpointed segments
# are recomputed during backward.  The recomputed layers will append NEW tensors
# to these lists, but by that point we have already summed and added the ORIGINAL
# tensors to the total loss.  The recomputed orphan tensors are simply ignored.
# Gradient flow through the original tensors works correctly because gradient
# checkpointing re-executes the segment on demand when autograd traverses the
# saved computation graph.
_global_moe_aux_losses: list = []

# Diversity loss accumulator (separate from balance loss so the two can be
# weighted and logged independently).
_global_moe_diversity_losses: list = []

# Detached, low-overhead training diagnostics. ``current`` belongs to one
# original model forward; it is committed immediately after that forward.
# This two-stage design prevents gradient-checkpoint recomputation during
# backward from being counted a second time.
_global_moe_diagnostics: dict = {
    "enabled": False,
    "current": {},
    "window": {},
}


def begin_moe_diagnostics(enabled: bool) -> None:
    """Start a fresh per-forward diagnostic capture."""
    _global_moe_diagnostics["current"].clear()
    _global_moe_diagnostics["enabled"] = bool(enabled)


def commit_moe_diagnostics() -> None:
    """Commit this forward's detached statistics to the logging window."""
    current = _global_moe_diagnostics["current"]
    window = _global_moe_diagnostics["window"]
    for key, values in current.items():
        if key not in window:
            window[key] = {name: value.clone() for name, value in values.items()}
        else:
            for name, value in values.items():
                window[key][name].add_(value)
    current.clear()
    # Recomputed checkpoint segments run after this point. Keeping capture off
    # ensures those forwards cannot contaminate the next logging window.
    _global_moe_diagnostics["enabled"] = False


def _merge_serialized_diagnostics(items: List[dict]) -> dict:
    merged: dict = {}
    for item in items:
        for key, values in item.items():
            if key not in merged:
                merged[key] = {
                    name: torch.as_tensor(value, dtype=torch.float64)
                    for name, value in values.items()
                }
            else:
                for name, value in values.items():
                    merged[key][name] += torch.as_tensor(value, dtype=torch.float64)
    return merged


def collect_moe_diagnostic_metrics() -> Dict[str, float]:
    """Pop, DDP-merge, and format the pending diagnostic window.

    Must be called collectively on every rank when distributed training is
    initialized. Only small CPU dictionaries are communicated.
    """
    window = _global_moe_diagnostics["window"]
    local = {
        key: {name: value.detach().double().cpu().tolist()
              for name, value in values.items()}
        for key, values in window.items()
    }
    window.clear()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local)
        merged = _merge_serialized_diagnostics(gathered)
    else:
        merged = _merge_serialized_diagnostics([local])

    metrics: Dict[str, float] = {}
    eps = 1e-12
    for key in sorted(merged):
        stat = merged[key]
        count = float(stat["token_count"].item())
        if count <= 0:
            continue
        prefix = f"moe_diag/{key}"
        argmax_counts = stat["argmax_counts"]
        probability_sums = stat["probability_sums"]
        expert_sq = stat["expert_sq"]
        num_experts = int(argmax_counts.numel())

        metrics[f"{prefix}/tokens"] = count
        metrics[f"{prefix}/router_entropy"] = float(stat["entropy_sum"].item() / count)
        metrics[f"{prefix}/router_entropy_normalized"] = float(
            stat["entropy_sum"].item() / (count * math.log(max(2, num_experts)))
        )
        metrics[f"{prefix}/router_max_probability"] = float(
            stat["max_probability_sum"].item() / count
        )
        element_count = max(1.0, float(stat["element_count"].item()))
        for i in range(num_experts):
            metrics[f"{prefix}/expert_{i}_argmax_fraction"] = float(argmax_counts[i].item() / count)
            metrics[f"{prefix}/expert_{i}_mean_probability"] = float(probability_sums[i].item() / count)
            metrics[f"{prefix}/expert_{i}_residual_norm"] = math.sqrt(
                max(0.0, float(expert_sq[i].item()))
            )
            metrics[f"{prefix}/expert_{i}_residual_rms"] = math.sqrt(
                max(0.0, float(expert_sq[i].item())) / element_count
            )

        base_sq = max(eps, float(stat["base_sq"].item()))
        metrics[f"{prefix}/delta_to_base_norm"] = math.sqrt(
            max(0.0, float(stat["delta_sq"].item())) / base_sq
        )
        pair_dot = stat["pair_dot"]
        pair_idx = 0
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                denom = math.sqrt(max(eps, float(expert_sq[i].item() * expert_sq[j].item())))
                metrics[f"{prefix}/expert_{i}_{j}_residual_cosine"] = float(
                    pair_dot[pair_idx].item() / denom
                )
                pair_idx += 1
    return metrics


def clear_moe_aux_losses() -> None:
    """Clear accumulated balance aux losses. Call before self.model(...)."""
    _global_moe_aux_losses.clear()


def collect_moe_aux_losses() -> "torch.Tensor | None":
    """Sum and return all accumulated balance aux losses. Returns None if empty."""
    if not _global_moe_aux_losses:
        return None
    total = _global_moe_aux_losses[0]
    for t in _global_moe_aux_losses[1:]:
        total = total + t
    _global_moe_aux_losses.clear()
    return total


def clear_moe_diversity_losses() -> None:
    """Clear accumulated diversity losses. Call before self.model(...)."""
    _global_moe_diversity_losses.clear()


def collect_moe_diversity_losses() -> "torch.Tensor | None":
    """Sum and return all accumulated diversity losses. Returns None if empty."""
    if not _global_moe_diversity_losses:
        return None
    total = _global_moe_diversity_losses[0]
    for t in _global_moe_diversity_losses[1:]:
        total = total + t
    _global_moe_diversity_losses.clear()
    return total


def set_modality_ids(modality_ids: torch.Tensor) -> None:
    """Store ``modality_ids`` (batch, seq_len) in process-global state."""
    _global_moe_state["modality_ids"] = modality_ids


def get_modality_ids() -> "torch.Tensor | None":
    """Return the current ``modality_ids`` tensor, or None if not set."""
    return _global_moe_state["modality_ids"]


def clear_modality_ids() -> None:
    """Clear process-global modality_ids."""
    _global_moe_state["modality_ids"] = None


# ---------------------------------------------------------------------------
# Balance loss computation
# ---------------------------------------------------------------------------

def _compute_balance_loss(
    router_logits: torch.Tensor,
    router_weights: torch.Tensor,
    num_experts: int,
    loss_type: str,
) -> "torch.Tensor | None":
    """Compute per-layer load-balancing auxiliary loss.

    Args:
        router_logits:  (N, E) raw logits before softmax.
        router_weights: (N, E) softmax weights (float32).
        num_experts:    E.
        loss_type:      One of "none" / "load_variance" / "switch" /
                        "entropy" / "z_loss" / "switch+z_loss".

    Returns:
        Scalar tensor or None.
    """
    N = router_weights.shape[0]
    if loss_type == "none" or N == 0:
        return None
    loss = None

    # ── load_variance ────────────────────────────────────────────────────────
    # Minimize variance of mean expert utilization.
    # Perfect balance: mean_load = [1/E, …] → var = 0.
    if "load_variance" in loss_type:
        mean_load = router_weights.mean(dim=0)          # (E,)
        loss = mean_load.var()

    # ── switch ───────────────────────────────────────────────────────────────
    # Switch Transformer auxiliary loss: E * Σ_i f_i * P_i
    # f_i: argmax fraction (stop-gradient),  P_i: mean soft probability.
    if "switch" in loss_type:
        assignments = router_weights.argmax(dim=-1)     # (N,)
        f = torch.stack(
            [(assignments == i).float().mean() for i in range(num_experts)]
        ).detach()                                      # (E,)  no grad
        P = router_weights.mean(dim=0)                  # (E,)
        l = (num_experts * f * P).sum()
        loss = l if loss is None else loss + l

    # ── entropy ──────────────────────────────────────────────────────────────
    # Maximize per-token routing entropy (→ uniform distribution over experts).
    # Returns the *negative* entropy so it can be minimised like other losses.
    if "entropy" in loss_type:
        ent = -(router_weights * (router_weights + 1e-9).log()).sum(dim=-1).mean()
        l = -ent
        loss = l if loss is None else loss + l

    # ── z_loss ───────────────────────────────────────────────────────────────
    # Penalise large router logits to prevent over-confident (collapsed) routing.
    # ST-MoE paper recommends weight ≈ 1e-3 (much smaller than main balance loss).
    if "z_loss" in loss_type:
        l = router_logits.logsumexp(dim=-1).pow(2).mean()
        loss = l if loss is None else loss + l

    return loss


# ---------------------------------------------------------------------------
# Expert modules
# ---------------------------------------------------------------------------

class UpDownExpert(nn.Module):
    """Standard LoRA residual:  delta = B(A(x))."""

    def __init__(self, hidden_size: int, r: int):
        super().__init__()
        self.A = nn.Linear(hidden_size, r, bias=False)
        self.B = nn.Linear(r, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x))


class GateUpDownExpert(nn.Module):
    """SwiGLU-style LoRA residual:  delta = down(act(gate(x)) * up(x)).

    Mirrors the Qwen3_5MLP structure (gate+up → element-wise product → down)
    at a reduced rank r.  Reuses the activation function from the base MLP.
    """

    def __init__(self, hidden_size: int, r: int, act_fn):
        super().__init__()
        self.gate = nn.Linear(hidden_size, r, bias=False)
        self.up   = nn.Linear(hidden_size, r, bias=False)
        self.down = nn.Linear(r, hidden_size, bias=False)
        self.act_fn = act_fn
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up.weight,   a=math.sqrt(5))
        nn.init.zeros_(self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act_fn(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# SharedDenseFFN (MSC ablation: no modality routing)
# ---------------------------------------------------------------------------

class SharedDenseFFN(nn.Module):
    """Frozen base MLP + a single shared LoRA residual on all tokens.

    Ablation counterpart to ``ModalityAwareMoEFFN``: same expert architecture
    and layer mounting, but no modality hard-route / soft-router — one expert
    applies to the full ``(B, S, H)`` sequence.
    """

    def __init__(
        self,
        base_mlp: nn.Module,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        proj_mode: str = "up_down",
    ):
        super().__init__()

        assert proj_mode in ("up_down", "gate_up_down"), \
            f"proj_mode must be 'up_down' or 'gate_up_down', got '{proj_mode}'"

        self.base_mlp = base_mlp
        for p in self.base_mlp.parameters():
            p.requires_grad = False

        self.r = r
        self.scaling = lora_alpha / r
        self.proj_mode = proj_mode
        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )

        hidden_size: int = base_mlp.gate_proj.in_features
        act_fn = getattr(base_mlp, 'act_fn', nn.SiLU())

        if proj_mode == "gate_up_down":
            self.moe_shared_expert = GateUpDownExpert(hidden_size, r, act_fn)
        else:
            self.moe_shared_expert = UpDownExpert(hidden_size, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)

        Returns:
            (batch, seq_len, hidden_size) = base_mlp(x) + shared LoRA delta
        """
        base_out = self.base_mlp(x)
        delta = self.moe_shared_expert(self.lora_dropout(x)) * self.scaling
        return base_out + delta


# ---------------------------------------------------------------------------
# SharedPoolMoEFFN
# ---------------------------------------------------------------------------

class SharedPoolMoEFFN(nn.Module):
    """Frozen base MLP plus one router/expert pool shared by target modalities.

    Modality ids are used only to select eligible tokens.  The router receives
    hidden states alone, so it has no explicit modality-id input.  Text, RGB,
    speech, and any modality outside ``modal_list`` use only ``base_mlp``.
    """

    def __init__(
        self,
        base_mlp: nn.Module,
        modal_list: list,
        num_experts: int = 3,
        r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.0,
        proj_mode: str = "gate_up_down",
        balance_loss_type: str = "none",
    ):
        super().__init__()
        if proj_mode not in ("up_down", "gate_up_down"):
            raise ValueError(f"unsupported proj_mode: {proj_mode}")
        unknown = sorted(set(modal_list) - set(MODALITY_ID_MAP))
        if unknown:
            raise ValueError(f"unknown shared-pool modalities: {unknown}")
        if not modal_list:
            raise ValueError("SharedPoolMoEFFN requires at least one modality")
        if num_experts < 1:
            raise ValueError("SharedPoolMoEFFN requires num_experts >= 1")

        self.base_mlp = base_mlp
        for parameter in self.base_mlp.parameters():
            parameter.requires_grad = False

        self.modal_list = list(modal_list)
        self.modal_ids = tuple(MODALITY_ID_MAP[name] for name in self.modal_list)
        self.num_experts = num_experts
        self.r = r
        self.scaling = lora_alpha / r
        self.proj_mode = proj_mode
        self.balance_loss_type = balance_loss_type
        self.diagnostic_layer_idx = -1
        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )

        hidden_size: int = base_mlp.gate_proj.in_features
        act_fn = getattr(base_mlp, 'act_fn', nn.SiLU())
        self.moe_shared_router = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.kaiming_uniform_(self.moe_shared_router.weight, a=math.sqrt(5))
        self.moe_shared_experts = nn.ModuleList()
        for _ in range(num_experts):
            if proj_mode == "gate_up_down":
                expert = GateUpDownExpert(hidden_size, r, act_fn)
            else:
                expert = UpDownExpert(hidden_size, r)
            self.moe_shared_experts.append(expert)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_mlp(x)
        modality_ids = get_modality_ids()
        if modality_ids is None:
            return base_out

        shared_mask = torch.zeros_like(modality_ids, dtype=torch.bool)
        for modality_id in self.modal_ids:
            shared_mask |= modality_ids == modality_id

        if not shared_mask.any():
            if self.training:
                dummy = torch.zeros(1, x.shape[-1], dtype=x.dtype, device=x.device)
                base_out = base_out + 0.0 * self.moe_shared_router(dummy).sum()
                for expert in self.moe_shared_experts:
                    base_out = base_out + 0.0 * expert(dummy).sum()
            return base_out

        tokens = x[shared_mask]
        router_logits = self.moe_shared_router(tokens)
        router_weights = F.softmax(router_logits.float(), dim=-1).to(tokens.dtype)

        if self.training and self.balance_loss_type != "none":
            balance_loss = _compute_balance_loss(
                router_logits.float(),
                router_weights.float(),
                self.num_experts,
                self.balance_loss_type,
            )
            if balance_loss is not None:
                _global_moe_aux_losses.append(balance_loss)

        expert_out = torch.zeros(
            tokens.shape[0], base_out.shape[-1],
            dtype=base_out.dtype, device=base_out.device,
        )
        expert_deltas = []
        for index, expert in enumerate(self.moe_shared_experts):
            expert_delta = expert(self.lora_dropout(tokens))
            expert_out = (
                expert_out
                + router_weights[:, index:index + 1] * expert_delta
            )
            if self.training:
                expert_deltas.append(expert_delta)

        delta = torch.zeros_like(base_out)
        delta[shared_mask] = expert_out * self.scaling

        if self.training and _global_moe_diagnostics["enabled"]:
            with torch.no_grad():
                weights_float = router_weights.float()
                assignments = weights_float.argmax(dim=-1)
                scaling_sq = self.scaling * self.scaling
                expert_sq = torch.stack([
                    item.detach().float().square().sum() * scaling_sq
                    for item in expert_deltas
                ])
                pair_dot = []
                for first in range(self.num_experts):
                    for second in range(first + 1, self.num_experts):
                        pair_dot.append(
                            (expert_deltas[first].detach().float()
                             * expert_deltas[second].detach().float()).sum()
                            * scaling_sq
                        )
                key = f"layer_{self.diagnostic_layer_idx}/shared"
                _global_moe_diagnostics["current"][key] = {
                    "token_count": torch.tensor(float(tokens.shape[0]), device=x.device),
                    "element_count": torch.tensor(float(tokens.numel()), device=x.device),
                    "argmax_counts": torch.stack([
                        (assignments == index).sum().float()
                        for index in range(self.num_experts)
                    ]),
                    "probability_sums": weights_float.sum(dim=0),
                    "entropy_sum": -(weights_float * (weights_float + 1e-9).log()).sum(),
                    "max_probability_sum": weights_float.max(dim=-1).values.sum(),
                    "base_sq": base_out[shared_mask].detach().float().square().sum(),
                    "delta_sq": (expert_out.detach().float() * self.scaling).square().sum(),
                    "expert_sq": expert_sq,
                    "pair_dot": (
                        torch.stack(pair_dot)
                        if pair_dot else torch.empty(0, device=x.device)
                    ),
                }

        if self.training and len(expert_deltas) > 1:
            prototypes = []
            for index, expert_delta in enumerate(expert_deltas):
                responsibility = router_weights[:, index]
                prototypes.append(
                    (responsibility.unsqueeze(-1) * expert_delta).sum(0)
                    / (responsibility.sum() + 1e-8)
                )
            prototype_stack = torch.stack(prototypes, dim=0)
            prototype_norms = prototype_stack.norm(dim=-1)
            with torch.no_grad():
                norm_gate = (prototype_norms.min() > 1e-2).float()
            normalized = F.normalize(
                prototype_stack.float(), p=2, dim=-1, eps=1e-8)
            similarity = normalized @ normalized.T
            off_diagonal = ~torch.eye(
                self.num_experts, dtype=torch.bool, device=similarity.device)
            diversity_loss = similarity[off_diagonal].abs().mean() * norm_gate
            _global_moe_diversity_losses.append(diversity_loss)

        return base_out + delta


# ---------------------------------------------------------------------------
# ModalityAwareMoEFFN
# ---------------------------------------------------------------------------

class ModalityAwareMoEFFN(nn.Module):
    """Frozen base MLP + per-modality LoRA MoE residual.

    Args:
        base_mlp:              The original ``Qwen3_5MLP`` (will be frozen).
        modal_list:            Modalities that have LoRA experts, e.g. ['depth'].
        num_experts_per_modal: Number of LoRA experts per modality (default 2).
        r:                     LoRA rank.
        lora_alpha:            LoRA scaling = lora_alpha / r.
        lora_dropout:          Dropout applied to expert inputs.
        proj_mode:             Expert architecture:
                               - "up_down"      : B(A(x))  [2 matrices/expert]
                               - "gate_up_down" : down(act(gate(x))*up(x))  [3 matrices/expert]
    """

    def __init__(
        self,
        base_mlp: nn.Module,
        modal_list: list,
        num_experts_per_modal: int = 2,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        proj_mode: str = "up_down",
        balance_loss_type: str = "none",
    ):
        super().__init__()

        assert proj_mode in ("up_down", "gate_up_down"), \
            f"proj_mode must be 'up_down' or 'gate_up_down', got '{proj_mode}'"

        # ── Frozen base MLP ────────────────────────────────────────────────
        self.base_mlp = base_mlp
        for p in self.base_mlp.parameters():
            p.requires_grad = False

        self.modal_list            = list(modal_list)
        self.num_experts_per_modal = num_experts_per_modal
        self.r                     = r
        self.scaling               = lora_alpha / r
        self.proj_mode             = proj_mode
        self.balance_loss_type     = balance_loss_type
        # Set to the actual decoder index by apply_moe_ffn_surgery().
        self.diagnostic_layer_idx  = -1
        self.lora_dropout          = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )

        # Infer hidden_size: Qwen3_5MLP.gate_proj = Linear(hidden_size, intermediate_size)
        hidden_size: int = base_mlp.gate_proj.in_features

        # Activation function for gate_up_down mode
        act_fn = getattr(base_mlp, 'act_fn', nn.SiLU())

        # ── Per-modality routers + expert pools ────────────────────────────
        self.moe_routers = nn.ModuleDict()   # modal → Linear(hidden, num_experts)
        self.moe_experts = nn.ModuleDict()   # modal → ModuleList[Expert]

        for m in self.modal_list:
            router = nn.Linear(hidden_size, num_experts_per_modal, bias=False)
            nn.init.kaiming_uniform_(router.weight, a=math.sqrt(5))
            self.moe_routers[m] = router

            experts = nn.ModuleList()
            for _ in range(num_experts_per_modal):
                if proj_mode == "gate_up_down":
                    experts.append(GateUpDownExpert(hidden_size, r, act_fn))
                else:
                    experts.append(UpDownExpert(hidden_size, r))
            self.moe_experts[m] = experts

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)

        Returns:
            (batch, seq_len, hidden_size) = base_mlp(x) + modality LoRA delta
        """
        base_out = self.base_mlp(x)

        modality_ids = get_modality_ids()
        if modality_ids is None:
            return base_out

        delta = torch.zeros_like(base_out)

        for m in self.modal_list:
            m_id = MODALITY_ID_MAP[m]
            mask = (modality_ids == m_id)   # (batch, seq_len)

            if not mask.any():
                # During training, keep expert params in the autograd graph (ZeRO-2
                # AllReduce requirement): add 0.0 * expert(zeros).sum() so that every
                # expert parameter appears in at least one rank's compute graph.
                # Use a zero vector (not actual hidden state) to avoid float16 overflow:
                # gate(x)*up(x) squares the magnitude, and sum() over 2048 elements can
                # push the result past float16 max (65504) → inf → 0.0*inf = NaN.
                # During inference, skip entirely — no autograd needed.
                if self.training:
                    dummy = torch.zeros(1, x.shape[-1], dtype=x.dtype, device=x.device)
                    for expert in self.moe_experts[m]:
                        base_out = base_out + 0.0 * expert(dummy).sum()
                    base_out = base_out + 0.0 * self.moe_routers[m](dummy).sum()
                continue

            # ── Step 1: hard-route — extract tokens of this modality ──────
            tokens = x[mask]                                        # (N, hidden)

            # ── Step 2: soft-route within the expert pool ─────────────────
            router_logits  = self.moe_routers[m](tokens)           # (N, num_experts)
            router_weights = F.softmax(
                router_logits.float(), dim=-1
            ).to(tokens.dtype)                                      # (N, num_experts)

            # ── Step 2b: accumulate balance loss (training only) ──────────
            if self.training and self.balance_loss_type != "none":
                bal_loss = _compute_balance_loss(
                    router_logits.float(),
                    router_weights.float(),
                    self.num_experts_per_modal,
                    self.balance_loss_type,
                )
                if bal_loss is not None:
                    _global_moe_aux_losses.append(bal_loss)

            # ── Step 3: weighted sum of expert outputs ────────────────────
            # Keep individual expert deltas (needed for diversity loss below).
            expert_out = torch.zeros(
                tokens.shape[0], base_out.shape[-1],
                dtype=base_out.dtype, device=base_out.device,
            )
            expert_deltas: list = []   # List of (N, hidden) tensors, one per expert
            for i, expert in enumerate(self.moe_experts[m]):
                lora_out = expert(self.lora_dropout(tokens))        # (N, hidden)
                expert_out = expert_out + router_weights[:, i:i+1] * lora_out
                if self.training:
                    expert_deltas.append(lora_out)

            delta[mask] = expert_out * self.scaling

            # Detached sufficient statistics only; no large activation is kept.
            # Expert residuals include LoRA scaling so their RMS is directly
            # comparable to the residual actually added to the frozen MLP.
            if self.training and _global_moe_diagnostics["enabled"]:
                with torch.no_grad():
                    rw = router_weights.float()
                    assignments = rw.argmax(dim=-1)
                    scaling_sq = self.scaling * self.scaling
                    expert_sq = torch.stack([
                        d.detach().float().square().sum() * scaling_sq
                        for d in expert_deltas
                    ])
                    pair_dot = []
                    for i in range(self.num_experts_per_modal):
                        for j in range(i + 1, self.num_experts_per_modal):
                            pair_dot.append(
                                (expert_deltas[i].detach().float()
                                 * expert_deltas[j].detach().float()).sum() * scaling_sq
                            )
                    key = f"layer_{self.diagnostic_layer_idx}/{m}"
                    _global_moe_diagnostics["current"][key] = {
                        "token_count": torch.tensor(float(tokens.shape[0]), device=x.device),
                        "element_count": torch.tensor(float(tokens.numel()), device=x.device),
                        "argmax_counts": torch.stack([
                            (assignments == i).sum().float()
                            for i in range(self.num_experts_per_modal)
                        ]),
                        "probability_sums": rw.sum(dim=0),
                        "entropy_sum": -(rw * (rw + 1e-9).log()).sum(),
                        "max_probability_sum": rw.max(dim=-1).values.sum(),
                        "base_sq": base_out[mask].detach().float().square().sum(),
                        "delta_sq": (expert_out.detach().float() * self.scaling).square().sum(),
                        "expert_sq": expert_sq,
                        "pair_dot": (
                            torch.stack(pair_dot)
                            if pair_dot else torch.empty(0, device=x.device)
                        ),
                    }

            # ── Step 3c: expert diversity loss (training only) ────────────
            # Goal: push different experts to specialise in different directions,
            # preventing them from collapsing to the same transformation.
            #
            # Method:
            #   For each expert i, compute a *responsibility-weighted prototype*:
            #     proto_i = Σ_t  router_weight_{t,i} * expert_delta_i(x_t)
            #               ────────────────────────────────────────────────
            #                        Σ_t  router_weight_{t,i}
            #   Then minimise the mean pairwise cosine similarity of proto vectors.
            #
            # Why residual delta, not full FFN output:
            #   base_mlp output is identical for all experts (frozen).  Only the
            #   expert delta captures what each expert *uniquely* contributes.
            #
            # Warmup: norm guard skips the loss while B matrices are still near
            #   zero (standard LoRA zero-init for B).  Once experts have non-zero
            #   outputs the loss activates naturally.  An additional linear warmup
            #   factor is applied by UnisenseQwen35ForCausalLM.forward() based on a
            #   training-forward-pass counter, giving a smooth ramp-up.
            if self.training and len(expert_deltas) > 1:
                prototypes = []
                for i, delta_i in enumerate(expert_deltas):
                    w_i = router_weights[:, i]            # (N,), keeps grad
                    w_sum = w_i.sum() + 1e-8
                    proto = (w_i.unsqueeze(-1) * delta_i).sum(0) / w_sum  # (hidden,)
                    prototypes.append(proto)

                proto_stack = torch.stack(prototypes, dim=0)        # (E, hidden)
                proto_norms = proto_stack.norm(dim=-1)              # (E,)

                # Always compute diversity loss with the same code path in both
                # the original forward and gradient-checkpoint recompute.
                # A conditional branch (if proto_norms.min() > 1e-4) causes
                # CheckpointError because CUDA reductions are non-deterministic:
                # the comparison can flip at the boundary between forward and
                # recompute, yielding a different number of save_for_backward
                # calls (3 saves × N_moe_layers discrepancy).
                # Fix: use a detached scalar gate so the autograd graph shape is
                # identical in both passes; F.normalize(eps=1e-8) handles
                # near-zero prototypes safely (no inf/NaN).
                with torch.no_grad():
                    # Threshold raised to 1e-2 (from 1e-4): F.normalize backward
                    # amplifies gradients by 1/||x||; at ||x||≈1e-4 that is ×10000,
                    # which causes NaN in bfloat16.  1e-2 ensures prototypes have
                    # meaningful magnitude before the loss activates.
                    norm_gate = (proto_norms.min() > 1e-2).float()  # 0.0 or 1.0
                # Cast to float32 before normalize: bfloat16 F.normalize backward
                # amplifies small-norm gradients into NaN (Bug 9).
                proto_normed = F.normalize(proto_stack.float(), p=2, dim=-1, eps=1e-8)
                sim_mat = proto_normed @ proto_normed.T              # (E, E) float32
                eye_mask = ~torch.eye(
                    self.num_experts_per_modal,
                    dtype=torch.bool,
                    device=sim_mat.device,
                )
                # abs(): penalise both over-alignment and anti-alignment.
                # norm_gate zeros the loss during early training (B matrices
                # near zero) without branching.
                div_loss = sim_mat[eye_mask].abs().mean() * norm_gate
                _global_moe_diversity_losses.append(div_loss)

        return base_out + delta


# ---------------------------------------------------------------------------
# Layer index computation
# ---------------------------------------------------------------------------

def _split_into_groups(layer_types: List[str]) -> List[List[int]]:
    """将 layer_types 按 full_attention 边界划分为 groups。

    每个 group 是以一个 full_attention 层结尾的连续层块。
    Qwen3.5-2B 示例（24 层）：
        group 0 → [0, 1, 2, 3]   (3×linear + 1×full)
        group 1 → [4, 5, 6, 7]
        ...
        group 5 → [20, 21, 22, 23]
    """
    groups: List[List[int]] = []
    current: List[int] = []
    for i, t in enumerate(layer_types):
        current.append(i)
        if t == "full_attention":
            groups.append(current)
            current = []
    if current:
        # 末尾没有 full_attention 收尾的层（不常见），也作为一组
        groups.append(current)
    return groups


def get_moe_layer_indices(
    layer_types: List[str],
    full_attn_only: bool,
    group_indices: List[int],
) -> List[int]:
    """Compute which LLM layer indices to apply MoE to.

    Qwen3.5-2B has 24 layers with the repeating pattern:
        [linear, linear, linear, full] × 6  →  6 "groups" (numbered 0-5)

    A *group* is defined as a contiguous block of layers whose last layer
    is a ``full_attention`` layer.

    Args:
        layer_types:    List of "full_attention" / "linear_attention" strings
                        (from ``config.text_config.layer_types``).
        full_attn_only: If True  → only patch the full_attention layers
                                   in the selected groups.
                        If False → patch every layer in the selected groups.
        group_indices:  Explicit list of group indices (0-based) to patch.
                        E.g. [0, 2, 5] selects groups 0, 2, and 5.

    Returns:
        Sorted list of layer indices.

    Examples (Qwen3.5-2B, pattern lin×3+full, 6 groups):
        group_indices=[0, 1],      full_attn_only=True  → [3, 7]
        group_indices=[0, 1],      full_attn_only=False → [0,1,2,3,4,5,6,7]
        group_indices=[0],         full_attn_only=True  → [3]
        group_indices=[0],         full_attn_only=False → [0,1,2,3]
        group_indices=[0, 2, 5],   full_attn_only=True  → [3, 11, 23]
        group_indices=[0, 2, 5],   full_attn_only=False → [0,1,2,3,8,9,10,11,20,21,22,23]
    """
    groups = _split_into_groups(layer_types)

    if not groups:
        raise ValueError("No full_attention layers found in layer_types.")

    invalid = [g for g in group_indices if g < 0 or g >= len(groups)]
    if invalid:
        raise ValueError(
            f"group_indices {invalid} out of range (model has {len(groups)} groups, 0-{len(groups)-1})."
        )

    result: List[int] = []
    for g in group_indices:
        for layer_idx in groups[g]:
            if full_attn_only and layer_types[layer_idx] != "full_attention":
                continue
            result.append(layer_idx)

    return sorted(set(result))


# ---------------------------------------------------------------------------
# Model surgery helper
# ---------------------------------------------------------------------------

def apply_moe_ffn_surgery(
    unisense_model,
    modal_list: List[str],
    num_experts_per_modal: int = 2,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    proj_mode: str = "up_down",
    full_attn_only: bool = True,
    balance_loss_type: str = "none",
    group_indices: Optional[List[int]] = None,
    shared_dense: bool = False,
    shared_pool: bool = False,
) -> None:
    """Replace ``.mlp`` of selected transformer layers with MoE / SharedDense FFN.

    Args:
        unisense_model:            ``UnisenseQwen3_5_VLModel`` (i.e. ``model.get_model()``).
        modal_list:            Modalities to create experts for, e.g. ['depth'].
                               Ignored when ``shared_dense=True``.
        num_experts_per_modal: Experts per modality (default 2).
                               Ignored when ``shared_dense=True``.
        r:                     LoRA rank.
        lora_alpha:            LoRA alpha.
        lora_dropout:          Dropout on expert inputs.
        proj_mode:             "up_down" or "gate_up_down" — expert architecture.
        full_attn_only:        If True, only patch full_attention layers;
                               if False, patch all layers in the selected groups.
        balance_loss_type:     Aux load-balance loss type (see moe_ffn.py).
                               Ignored when ``shared_dense=True``.
        group_indices:         List of group indices (0-based) to patch.
                               Qwen3.5-2B has 6 groups (0-5).
                               Defaults to [0] (first group only) if not provided.
        shared_dense:          If True, install ``SharedDenseFFN`` (single shared
                               LoRA on all tokens) instead of ``ModalityAwareMoEFFN``.
        shared_pool:           If True, install ``SharedPoolMoEFFN`` (one router
                               and expert pool shared by all listed modalities).

    The layer_types list is read from ``unisense_model.config.text_config.layer_types``.
    """
    if group_indices is None:
        group_indices = [0]
    if shared_dense and shared_pool:
        raise ValueError("shared_dense and shared_pool are mutually exclusive")

    # Read layer_types from config
    text_config  = getattr(unisense_model.config, 'text_config', None)
    layer_types  = getattr(text_config, 'layer_types', None) if text_config else None
    if layer_types is None:
        raise ValueError(
            "unisense_model.config.text_config.layer_types not found. "
            "Cannot determine MoE layer indices automatically."
        )

    layer_indices = get_moe_layer_indices(layer_types, full_attn_only, group_indices)

    layers = unisense_model.language_model.layers

    # A modality-only FFN residual on the final decoder layer cannot influence
    # supervised text tokens: it changes only modality-token states, and there
    # is no later attention layer through which those states can reach text.
    # This used to create a permanently zero-initialized expert pool at layer
    # N-1 (verified across depth and pointcloud checkpoints).  Keep the final
    # layer for SharedDense/text experts, because those directly modify text
    # token states before the LM head.
    final_layer_idx = len(layers) - 1
    if (not shared_dense and 'text' not in modal_list
            and final_layer_idx in layer_indices):
        layer_indices.remove(final_layer_idx)
        print(
            f"[MoE surgery] skipping final decoder layer {final_layer_idx}: "
            "a modality-only MLP residual has no downstream attention path "
            "to supervised text tokens"
        )
    if not layer_indices:
        raise ValueError(
            "MoE layer selection is empty after excluding the ineffective final "
            "decoder layer. Select at least one earlier group."
        )

    mode_str = (
        "shared_dense" if shared_dense
        else f"shared_pool={modal_list}" if shared_pool
        else f"modal_list={modal_list}"
    )
    print(
        f"[MoE surgery] {mode_str}, proj_mode={proj_mode}, "
        f"group_indices={group_indices}, full_attn_only={full_attn_only}, "
        f"target layers={layer_indices}"
    )

    for idx in layer_indices:
        if idx >= len(layers):
            raise ValueError(
                f"Layer index {idx} out of range (model has {len(layers)} layers)."
            )
        layer = layers[idx]
        original_mlp = layer.mlp

        if isinstance(
                original_mlp,
                (ModalityAwareMoEFFN, SharedDenseFFN, SharedPoolMoEFFN)):
            print(f"  Layer {idx}: already {type(original_mlp).__name__}, skipping.")
            continue

        layer_type = layer_types[idx]
        if shared_dense:
            moe_ffn = SharedDenseFFN(
                base_mlp=original_mlp,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                proj_mode=proj_mode,
            )
            layer.mlp = moe_ffn
            print(
                f"  Layer {idx} ({layer_type}): mlp → SharedDenseFFN"
                f"(r={r}, proj_mode={proj_mode})"
            )
        elif shared_pool:
            moe_ffn = SharedPoolMoEFFN(
                base_mlp=original_mlp,
                modal_list=modal_list,
                num_experts=num_experts_per_modal,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                proj_mode=proj_mode,
                balance_loss_type=balance_loss_type,
            )
            moe_ffn.diagnostic_layer_idx = idx
            layer.mlp = moe_ffn
            print(
                f"  Layer {idx} ({layer_type}): mlp → SharedPoolMoEFFN"
                f"(n_experts={num_experts_per_modal}, r={r}, "
                f"proj_mode={proj_mode}, balance={balance_loss_type})"
            )
        else:
            moe_ffn = ModalityAwareMoEFFN(
                base_mlp=original_mlp,
                modal_list=modal_list,
                num_experts_per_modal=num_experts_per_modal,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                proj_mode=proj_mode,
                balance_loss_type=balance_loss_type,
            )
            moe_ffn.diagnostic_layer_idx = idx
            layer.mlp = moe_ffn
            print(
                f"  Layer {idx} ({layer_type}): mlp → ModalityAwareMoEFFN"
                f"(n_experts={num_experts_per_modal}, r={r}, proj_mode={proj_mode}, "
                f"balance={balance_loss_type})"
            )
