#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
#    Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
#    Copyright 2024 Zhisheng Zhong, Chengyao Wang
# ------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.utils import logging
from transformers.generation.utils import GenerateOutput

try:
    from transformers import (
        Qwen3_5Config,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5Model,
    )
except ImportError:
    Qwen3_5Config = getattr(transformers, "Qwen3_5Config")
    Qwen3_5ForConditionalGeneration = getattr(transformers, "Qwen3_5ForConditionalGeneration")
    Qwen3_5Model = getattr(transformers, "Qwen3_5Model")

from unisense.model.unisense_arch_qwen import UnisenseMetaModel, UnisenseMetaForCausalLM
from unisense.model.moe_ffn import (
    set_modality_ids, clear_modality_ids, apply_moe_ffn_surgery,
    clear_moe_diversity_losses, collect_moe_diversity_losses,
    begin_moe_diagnostics, commit_moe_diagnostics,
)
from torch.nn import CrossEntropyLoss
from unisense.constants import IGNORE_INDEX


logger = logging.get_logger(__name__)

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class UnisenseCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class UnisenseQwen35Config(Qwen3_5Config):
    model_type = "Unisense_Qwen3_5_VL"

    def __init__(self, **kwargs):
        # Normalize metadata when explicitly loading an older base checkpoint.
        kwargs.pop("model_type", None)
        kwargs["architectures"] = ["UnisenseQwen35ForCausalLM"]
        super().__init__(**kwargs)
        # Qwen3.5 stores text dims under text_config in some versions;
        # expose flat aliases for legacy Unisense modules.
        if not hasattr(self, "hidden_size"):
            text_config = getattr(self, "text_config", None)
            if text_config is not None and hasattr(text_config, "hidden_size"):
                self.hidden_size = text_config.hidden_size
        if not hasattr(self, "vocab_size"):
            text_config = getattr(self, "text_config", None)
            if text_config is not None and hasattr(text_config, "vocab_size"):
                self.vocab_size = text_config.vocab_size


class UnisenseQwen3_5_VLModel(UnisenseMetaModel, Qwen3_5Model):
    config_class = UnisenseQwen35Config

    def __init__(self, config: Qwen3_5Config):
        super(UnisenseQwen3_5_VLModel, self).__init__(config)

    def embed_tokens(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        兼容 Unisense 多模态框架中对 `embed_tokens` 的调用。
        """
        input_embeddings = self.get_input_embeddings()
        return input_embeddings(input_ids)


class UnisenseQwen35ForCausalLM(Qwen3_5ForConditionalGeneration, UnisenseMetaForCausalLM):
    config_class = UnisenseQwen35Config
    # Loading an LLM-only checkpoint intentionally misses vision branch weights.
    _keys_to_ignore_on_load_missing = [r"^model\.visual\."]

    def __init__(self, config):
        # 跟随 LLaVA 风格，跳过父类头部定义，仅初始化基础模块
        super(Qwen3_5ForConditionalGeneration, self).__init__(config)
        self.model = UnisenseQwen3_5_VLModel(config)
        text_config = getattr(config, "text_config", None)
        hidden_size = getattr(config, "hidden_size", None)
        vocab_size = getattr(config, "vocab_size", None)
        if text_config is not None:
            hidden_size = hidden_size if hidden_size is not None else getattr(text_config, "hidden_size", None)
            vocab_size = vocab_size if vocab_size is not None else getattr(text_config, "vocab_size", None)
        if hidden_size is None or vocab_size is None:
            raise ValueError("Cannot infer hidden_size/vocab_size from config for UnisenseQwen35ForCausalLM")

        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Counter for training forward passes (labels present).
        # Used as a proxy for optimizer steps when computing the diversity-loss
        # warmup factor.  Incremented inside forward(); persists across steps.
        # Registered as a persistent buffer (not a plain Python int) so it is
        # included in state_dict() and survives checkpoint save/resume — a
        # plain attribute would silently reset to 0 on every resume, restarting
        # the diversity-loss warmup from scratch.
        self.register_buffer('_moe_forward_count', torch.zeros((), dtype=torch.long), persistent=True)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        speeches: Optional[torch.FloatTensor] = None,
        speeches_asr: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        caption: torch.LongTensor = None,
        objects_len=None,
        pointclouds=None,
        imus=None,
        depth_images=None,
        tactile_images=None,
        thermal_images=None,
        radars=None,
        smell_signals=None,
        motions=None,
        modality_instance_ids=None,
        target_modality_id=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        loss_align = None
        loss_depth_caption = None
        loss_pointcloud_caption = None
        loss_imu_caption = None
        loss_tactile_caption = None
        loss_thermal_caption = None
        loss_radar_caption = None
        loss_smell_caption = None
        loss_motion_caption = None
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                loss_align,
                loss_depth_caption,
                loss_pointcloud_caption,
                loss_imu_caption,
                loss_tactile_caption,
                loss_thermal_caption,
                loss_radar_caption,
                loss_smell_caption,
                loss_motion_caption,
                rope_deltas,
            ) = self.prepare_inputs_labels_for_text_image_speech_qwen2vl(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                speeches,
                speeches_asr,
                rope_deltas,
                caption,
                objects_len,
                pointclouds,
                imus,
                depth_images,
                tactile_images,
                thermal_images,
                radars,
                smell_signals,
                motions,
                modality_instance_ids,
                target_modality_id,
            )
        # Set (or clear) modality_ids in thread-local state so ModalityAwareMoEFFN
        # layers can hard-route tokens to the correct expert pool.
        # Always update — this also handles decode steps where prepare_inputs_labels
        # sets _pending_modality_ids=None (early return), ensuring stale prefill ids
        # from a previous step don't leak into the next step's MoE routing.
        _pending_mod_ids = getattr(self, '_pending_modality_ids', None)
        if _pending_mod_ids is not None:
            set_modality_ids(_pending_mod_ids)
        else:
            clear_modality_ids()

        # Reset aux-loss accumulators before the model forward so each step
        # starts clean (gradient-checkpointing recomputation may append extras
        # during backward, but we collect immediately after the forward below).
        # Increment training forward counter (used for diversity-loss warmup).
        from unisense.model.moe_ffn import clear_moe_aux_losses, collect_moe_aux_losses
        if self.training:
            clear_moe_aux_losses()
            clear_moe_diversity_losses()
            if labels is not None:
                self._moe_forward_count += 1
            diagnostic_interval = int(getattr(self.config, 'moe_diagnostic_interval', 0))
            grad_accum = max(1, int(getattr(
                self.config, 'moe_diagnostic_gradient_accumulation_steps', 1
            )))
            forward_count = int(self._moe_forward_count.item())
            optimizer_step = ((forward_count - 1) // grad_accum + 1) if forward_count > 0 else 0
            capture_diagnostics = (
                labels is not None and diagnostic_interval > 0
                and optimizer_step > 0 and optimizer_step % diagnostic_interval == 0
            )
            begin_moe_diagnostics(capture_diagnostics)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # NOTE: do NOT clear thread-local modality_ids here. Gradient checkpointing
        # recomputes the checkpointed layer segments during backward, AFTER this forward
        # call returns. If we clear here (e.g. in a finally block), the recomputation
        # will see None and take a different code path → CheckpointError (mismatched
        # saved tensor counts). Thread-local is updated at the TOP of each forward()
        # call (lines above), so stale ids from a previous step are always overwritten.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        if self.training:
            commit_moe_diagnostics()

        hidden_states = outputs[0]
        loss = None

        if labels is not None:
            # Training: only run lm_head on positions that actually contribute to
            # the CE loss (label != IGNORE_INDEX).  With Qwen3.5's 248K vocab the
            # full [B, L, vocab] logits tensor is huge (≈16GB at B=4/L=4096 fp32);
            # since SFT masks all prompt/visual tokens, the vast majority of those
            # positions are wasted.  Gathering valid hidden states *before* lm_head
            # avoids that allocation entirely.  Mathematically identical to the
            # full-logits + CrossEntropyLoss(ignore_index=-100) path.
            shift_hidden = hidden_states[..., :-1, :]      # (B, L-1, H)
            shift_labels = labels[..., 1:]                 # (B, L-1)
            valid_mask = shift_labels != IGNORE_INDEX      # (B, L-1)
            loss_fct = CrossEntropyLoss()
            if valid_mask.any():
                valid_hidden = shift_hidden[valid_mask]                       # (N_valid, H)
                valid_labels = shift_labels[valid_mask].to(valid_hidden.device)  # (N_valid,)
                valid_logits = self.lm_head(valid_hidden).float()            # (N_valid, vocab)
                loss = loss_fct(valid_logits, valid_labels)
            else:
                # No supervised tokens in this batch: keep lm_head in the autograd
                # graph with a zero-valued loss so DDP AllReduce buckets still fire.
                loss = self.lm_head(hidden_states[:, :1, :]).float().sum() * 0.0
            # Training never consumes the full logits tensor downstream.
            logits = None
        else:
            # Inference / generation: full logits are needed by the caller.
            logits = self.lm_head(hidden_states).float()

        if getattr(self.config, "asr_align", None) and self.training and loss_align is not None:
            if self.device.index == 0:
                if isinstance(loss_align, float):
                    print("ASR Align loss: {:.4f}, LLM CE loss: {:.4f}".format(loss_align, loss.item()))
                else:
                    print("ASR Align loss: {:.4f}, LLM CE loss: {:.4f}".format(loss_align.item(), loss.item()))
            loss += self.config.weight_lambda * loss_align

        if loss_depth_caption is not None and self.training:
            depth_caption_weight = getattr(self.config, 'depth_caption_weight', 0.01)
            depth_alignment = getattr(self.config, 'depth_alignment_type', 'none')
            if (self.device.index == 0 and depth_alignment != 'none'
                    and getattr(self.config, 'mm_depth_projector_type', None) is not None):
                loss_depth = loss_depth_caption if isinstance(loss_depth_caption, float) else loss_depth_caption.item()
                print("LLM CE loss: {:.4f}, Depth {} loss: {:.4f}".format(
                    loss.item(), depth_alignment, loss_depth))
            loss += depth_caption_weight * loss_depth_caption

        if loss_pointcloud_caption is not None and self.training:
            pc_caption_weight = getattr(self.config, 'pointcloud_caption_weight', 0.1)
            pc_alignment_name = getattr(
                self.config, 'pointcloud_alignment_type', 'none')
            if self.device.index == 0 and pc_alignment_name != 'none':
                loss_pc = loss_pointcloud_caption if isinstance(loss_pointcloud_caption, float) else loss_pointcloud_caption.item()
                print("LLM CE loss: {:.4f}, Pointcloud {} loss: {:.4f}".format(
                    loss.item(), pc_alignment_name, loss_pc))
            loss = loss + pc_caption_weight * loss_pointcloud_caption

        if loss_imu_caption is not None and self.training:
            imu_caption_weight = getattr(self.config, 'imu_caption_weight', 0.1)
            imu_alignment = getattr(self.config, 'imu_alignment_type', 'none')
            if self.device.index == 0 and imu_alignment != 'none':
                loss_imu = loss_imu_caption if isinstance(loss_imu_caption, float) else loss_imu_caption.item()
                print("LLM CE loss: {:.4f}, IMU {} loss: {:.4f}".format(
                    loss.item(), imu_alignment, loss_imu))
            loss = loss + imu_caption_weight * loss_imu_caption

        if loss_tactile_caption is not None and self.training:
            tactile_caption_weight = getattr(self.config, 'tactile_caption_weight', 0.1)
            tactile_alignment = getattr(self.config, 'tactile_alignment_type', 'none')
            if self.device.index == 0 and tactile_alignment != 'none':
                loss_tac = loss_tactile_caption if isinstance(loss_tactile_caption, float) else loss_tactile_caption.item()
                print("LLM CE loss: {:.4f}, Tactile {} loss: {:.4f}".format(
                    loss.item(), tactile_alignment, loss_tac))
            loss = loss + tactile_caption_weight * loss_tactile_caption

        if loss_thermal_caption is not None and self.training:
            thermal_caption_weight = getattr(self.config, 'thermal_caption_weight', 0.1)
            thermal_alignment = getattr(self.config, 'thermal_alignment_type', 'none')
            if self.device.index == 0 and thermal_alignment != 'none':
                loss_thr = loss_thermal_caption if isinstance(loss_thermal_caption, float) else loss_thermal_caption.item()
                print("LLM CE loss: {:.4f}, Thermal {} loss: {:.4f}".format(
                    loss.item(), thermal_alignment, loss_thr))
            loss = loss + thermal_caption_weight * loss_thermal_caption

        if loss_radar_caption is not None and self.training:
            radar_caption_weight = getattr(self.config, 'radar_caption_weight', 0.1)
            radar_alignment_name = getattr(
                self.config, 'radar_alignment_type', 'none')
            if self.device.index == 0 and radar_alignment_name != 'none':
                loss_rad = loss_radar_caption if isinstance(loss_radar_caption, float) else loss_radar_caption.item()
                print("LLM CE loss: {:.4f}, Radar {} loss: {:.4f}".format(
                    loss.item(), radar_alignment_name, loss_rad))
            loss = loss + radar_caption_weight * loss_radar_caption

        if loss_smell_caption is not None and self.training:
            smell_caption_weight = getattr(self.config, 'smell_caption_weight', 0.1)
            smell_alignment = getattr(self.config, 'smell_alignment_type', 'none')
            if self.device.index == 0 and smell_alignment != 'none':
                loss_sme = loss_smell_caption if isinstance(loss_smell_caption, float) else loss_smell_caption.item()
                print("LLM CE loss: {:.4f}, Smell {} loss: {:.4f}".format(
                    loss.item(), smell_alignment, loss_sme))
            loss = loss + smell_caption_weight * loss_smell_caption

        if loss_motion_caption is not None and self.training:
            motion_caption_weight = getattr(self.config, 'motion_caption_weight', 0.1)
            motion_alignment = getattr(self.config, 'motion_alignment_type', 'none')
            if self.device.index == 0 and motion_alignment != 'none':
                loss_mot = loss_motion_caption if isinstance(loss_motion_caption, float) else loss_motion_caption.item()
                print("LLM CE loss: {:.4f}, Motion {} loss: {:.4f}".format(
                    loss.item(), motion_alignment, loss_mot))
            loss = loss + motion_caption_weight * loss_motion_caption

        if self.training:
            forward_count = int(self._moe_forward_count.item())
            aux_start = int(getattr(
                self.config, 'moe_aux_start_forward', 0))
            moe_aux_active = forward_count > aux_start

            balance_loss_weight = getattr(self.config, 'moe_balance_loss_weight', 0.0)
            if balance_loss_weight > 0.0 and moe_aux_active:
                loss_balance = collect_moe_aux_losses()
                if loss_balance is not None:
                    if self.device.index == 0:
                        print("MoE balance loss: {:.4f} (×{})".format(
                            loss_balance.item(), balance_loss_weight))
                    loss = loss + balance_loss_weight * loss_balance

            diversity_loss_weight = getattr(self.config, 'moe_diversity_loss_weight', 0.0)
            if diversity_loss_weight > 0.0:
                loss_diversity = collect_moe_diversity_losses()
                if loss_diversity is not None:
                    # train.py converts the configured ratio over MoE-active
                    # optimizer steps to per-rank training forwards. The start
                    # offset is the phased-unfreeze boundary, so a frozen
                    # projector warm-up cannot consume this schedule.
                    # The norm guard in moe_ffn.py provides an additional safety
                    # net: it skips the loss while proto norms are below 1e-2.
                    warmup_forwards = int(getattr(
                        self.config, 'moe_diversity_warmup_forwards', 0))
                    warmup_progress = max(0, forward_count - aux_start)
                    if not moe_aux_active:
                        warmup_factor = 0.0
                    elif warmup_forwards > 0:
                        warmup_factor = min(
                            1.0, warmup_progress / warmup_forwards)
                    else:
                        warmup_factor = 1.0
                    if warmup_factor > 0.0:
                        if self.device.index == 0:
                            print(
                                "MoE diversity loss: {:.4f} (×{:.4f}, "
                                "warmup={:.3f}, active_forward={})".format(
                                    loss_diversity.item(), diversity_loss_weight,
                                    warmup_factor, warmup_progress))
                        loss = loss + diversity_loss_weight * warmup_factor * loss_diversity

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return UnisenseCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        speeches: Optional[torch.Tensor] = None,
        speeches_asr: Optional[torch.Tensor] = None,
        pointclouds=None,
        imus=None,
        depth_images=None,
        tactile_images=None,
        thermal_images=None,
        radars=None,
        smell_signals=None,
        motions=None,
        rope_deltas: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        caption = kwargs.pop("caption", None)
        objects_len = kwargs.pop("objects_len", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if (images is not None or speeches is not None
                or pointclouds is not None or imus is not None or depth_images is not None
                or tactile_images is not None or thermal_images is not None
                or radars is not None or smell_signals is not None
                or motions is not None):
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                rope_deltas,
            ) = self.prepare_inputs_labels_for_text_image_speech_qwen2vl(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                speeches,
                speeches_asr,
                rope_deltas,
                caption,
                objects_len,
                pointclouds,
                imus,
                depth_images,
                tactile_images,
                thermal_images,
                radars,
                smell_signals,
                motions,
            )

        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            # No multimodal input: clear any stale modality_ids from a previous call.
            # Without this, a prior prefill's _pending_modality_ids (wrong shape) would
            # be passed to MoE layers during the text-only prefill forward.
            self._pending_modality_ids = None

        # modality_ids are propagated automatically: self._pending_modality_ids was set
        # by prepare_inputs_labels above (prefill) or will be set/cleared inside each
        # forward() call (decode steps).  No manual thread-local management needed here.
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            rope_deltas=rope_deltas,
            **kwargs,
        )

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        speeches = kwargs.pop("speeches", None)
        speeches_asr = kwargs.pop("speeches_asr", None)
        pointclouds = kwargs.pop("pointclouds", None)
        imus = kwargs.pop("imus", None)
        radars = kwargs.pop("radars", None)
        smell_signals = kwargs.pop("smell_signals", None)
        motions = kwargs.pop("motions", None)
        rope_deltas = kwargs.pop("rope_deltas", None)
        caption = kwargs.pop("caption", None)
        objects_len = kwargs.pop("objects_len", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        # Remove Qwen3.5 native vision kwargs — Unisense handles vision via `images` argument.
        for _key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
            _inputs.pop(_key, None)
        if images is not None:
            _inputs["images"] = images
        if speeches is not None:
            _inputs["speeches"] = speeches
        if speeches_asr is not None:
            _inputs["speeches_asr"] = speeches_asr
        if pointclouds is not None:
            _inputs["pointclouds"] = pointclouds
        if imus is not None:
            _inputs["imus"] = imus
        if radars is not None:
            _inputs["radars"] = radars
        if smell_signals is not None:
            _inputs["smell_signals"] = smell_signals
        if motions is not None:
            _inputs["motions"] = motions
        if rope_deltas is not None:
            _inputs["rope_deltas"] = rope_deltas
        if caption is not None:
            _inputs["caption"] = caption
        if objects_len is not None:
            _inputs["objects_len"] = objects_len
        return _inputs


AutoConfig.register("Unisense_Qwen3_5_VL", UnisenseQwen35Config)
AutoModelForCausalLM.register(UnisenseQwen35Config, UnisenseQwen35ForCausalLM)
