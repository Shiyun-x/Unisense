import os
import inspect
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    logger,
)

# 兼容不同 transformers 版本：部分版本不再暴露 ALL_LAYERNORM_LAYERS
try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS  # type: ignore
except ImportError:
    # 回退到仅使用 torch.nn.LayerNorm（主要用于排除 weight decay）
    ALL_LAYERNORM_LAYERS = (nn.LayerNorm,)
from typing import List, Optional

from unisense.train.modality_sampler import TargetModalityBatchSampler


def maybe_zero_3(param, ignore_status=False, name=None):
    if hasattr(param, "ds_id"):
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(
    named_params, require_grad_only=True, named_buffers=None
):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        # Keep both C2 projection branches together in checkpoints, including
        # configurations that may explicitly freeze an alignment module.
        projection_head_markers = (
            '_alignment_student_head.', '_alignment_text_head.')
        to_return = {
            k: t for k, t in to_return.items()
            if t.requires_grad
            or any(marker in k for marker in projection_head_markers)
        }
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    # Normalization statistics are buffers, so named_parameters() omits them.
    if named_buffers is not None:
        for name, buffer in named_buffers:
            if name.endswith((
                'smell_tokenizer.channel_mean', 'smell_tokenizer.channel_std'
            )):
                to_return[name] = buffer.detach().cpu().clone()
    return to_return

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    
    megabatch_size = world_size * batch_size
    
    img_indices, img_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0 and isinstance(l, int)])
    img_shuffle = [img_indices[i] for i in get_length_grouped_indices(img_lengths, batch_size, world_size, generator=None)]
    img_megabatches = [img_shuffle[i : i + megabatch_size] for i in range(0, len(img_shuffle), megabatch_size)]
    
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]
    
    megabatches = img_megabatches[:-1] + lang_megabatches[:-1]
    
    if len([(i, l) for i, l in enumerate(lengths) if l > 0 and isinstance(l, float)]) > 0:
        spe_indices, spe_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0 and isinstance(l, float)])
        spe_lengths = (np.array(spe_lengths) - 0.5).astype(int).tolist()
        spe_shuffle = [spe_indices[i] for i in get_length_grouped_indices(spe_lengths, batch_size, world_size, generator=None)]
        spe_megabatches = [spe_shuffle[i : i + megabatch_size] for i in range(0, len(spe_shuffle), megabatch_size)]
        megabatches += spe_megabatches[:-1]
    
    # last_img = img_megabatches[-1]
    # last_lang = lang_megabatches[-1]
    # last_sp = spe_megabatches[-1]
    # additional_batch = last_mm + last_lang + last_sp
    
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # drop mix-modality data for stability
    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

class UnisenseTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        # transformers 新版 Trainer 用 processing_class 替代 tokenizer。
        # 这里做双向兼容，避免因为关键字变化导致初始化失败。
        trainer_init_params = set(inspect.signature(Trainer.__init__).parameters.keys())
        tokenizer = kwargs.get("tokenizer", None)

        if "tokenizer" not in trainer_init_params and tokenizer is not None:
            kwargs.pop("tokenizer", None)
            kwargs.setdefault("processing_class", tokenizer)

        if "processing_class" not in trainer_init_params and "processing_class" in kwargs:
            processing_class = kwargs.pop("processing_class")
            kwargs.setdefault("tokenizer", processing_class)

        # 先调用父类构造，创建好 accelerator 等组件
        super().__init__(*args, **kwargs)

        # 对外维持 self.tokenizer 访问习惯，兼容项目内旧逻辑。
        if tokenizer is None:
            tokenizer = getattr(self, "processing_class", None)
        if tokenizer is not None:
            self.tokenizer = tokenizer

        self._last_moe_diagnostic_collect_step = 0

        # 兼容旧版 accelerate：其 unwrap_model 不支持 keep_torch_compile 参数
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None:
            orig_unwrap = getattr(accelerator, "unwrap_model", None)
            if callable(orig_unwrap):
                def _unwrap_model_compatible(*u_args, **u_kwargs):
                    # 新版 transformers 会传 keep_torch_compile，我们在这里安全地吞掉
                    u_kwargs.pop("keep_torch_compile", None)
                    return orig_unwrap(*u_args, **u_kwargs)

                accelerator.unwrap_model = _unwrap_model_compatible

    def log(self, logs, start_time=None):
        """Merge pending MoE diagnostics into the normal Trainer log event."""
        config = getattr(self.model, "config", None)
        diagnostic_interval = int(getattr(config, "moe_diagnostic_interval", 0))
        current_step = int(getattr(self.state, "global_step", 0))
        crossed_diagnostic_step = (
            diagnostic_interval > 0
            and current_step // diagnostic_interval
            > self._last_moe_diagnostic_collect_step // diagnostic_interval
        )
        if crossed_diagnostic_step:
            from unisense.model.moe_ffn import collect_moe_diagnostic_metrics
            diagnostic_metrics = collect_moe_diagnostic_metrics()
            if diagnostic_metrics:
                logs = {**logs, **diagnostic_metrics}
            self._last_moe_diagnostic_collect_step = current_step

        # transformers changed this signature across releases.
        if "start_time" in inspect.signature(Trainer.log).parameters:
            return super().log(logs, start_time=start_time)
        return super().log(logs)

    # 兼容新版 transformers：_get_train_sampler 接收 train_dataset 参数
    def _get_train_sampler(self, train_dataset=None) -> Optional[torch.utils.data.Sampler]:
        # 如果上层没传入，则回退到实例上的属性，兼容旧版本调用方式
        if train_dataset is None:
            train_dataset = self.train_dataset

        if train_dataset is None or not has_length(train_dataset):
            return None

        if self.args.group_by_target_modality:
            if not hasattr(train_dataset, "target_modalities"):
                raise ValueError(
                    "group_by_target_modality requires dataset.target_modalities")
            return TargetModalityBatchSampler(
                modalities=train_dataset.target_modalities,
                per_device_batch_size=self.args.train_batch_size,
                world_size=self.args.world_size,
                seed=(
                    self.args.data_seed
                    if self.args.data_seed is not None else self.args.seed),
            )
        elif self.args.group_by_modality_length:
            lengths = train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.args.lr_multi is not None:
            lr_multi_dict = {}
            for _dict in self.args.lr_multi.split(','):
                _key_val = _dict.split(':')
                print("_key_val:", _key_val)
                lr_multi_dict[_key_val[0]] = float(_key_val[1])

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            elif self.args.lr_multi is not None:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and not any([_key in n for _key in lr_multi_dict.keys()]))
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and not any([_key in n for _key in lr_multi_dict.keys()]))
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                for _key in lr_multi_dict:
                    _is_moe_lr_group = _key in {
                        "moe_routers", "moe_experts", "moe_shared_expert",
                        "moe_shared_router", "moe_shared_experts",
                    }
                    _key_decay = [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and _key in n)
                        ]
                    _key_no_decay = [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and _key in n)
                        ]
                    print("Params LR Change:", _key, "NUM:", len(_key_decay), len(_key_no_decay))
                    if len(_key_decay) > 0:
                        optimizer_grouped_parameters.append(
                            {
                                "params": _key_decay,
                                "lr": self.args.learning_rate * lr_multi_dict[_key],
                                "weight_decay": self.args.weight_decay,
                                "unisense_moe_lr_group": _is_moe_lr_group,
                            },
                        )
                    if len(_key_no_decay) > 0:
                        optimizer_grouped_parameters.append(
                            {
                                "params": _key_no_decay,
                                "lr": self.args.learning_rate * lr_multi_dict[_key],
                                "weight_decay": 0.0,
                                "unisense_moe_lr_group": _is_moe_lr_group,
                            },
                        )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save vision Adapter
            keys_to_match = ['mm_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if is_main_process():
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        elif getattr(self.args, 'tune_speech_projector_only', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save speech Adapter
            keys_to_match = ['mm_speech_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if is_main_process():
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_speech_projector.bin'))
        else:
            if getattr(self.args, 'lora_enable', False):
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)

                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    self.model.base_model.named_parameters(),
                    named_buffers=self.model.base_model.named_buffers(),
                )
                if is_main_process():
                    self.model.config.save_pretrained(output_dir)
                    torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
            super(UnisenseTrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(UnisenseTrainer, self)._save(output_dir, state_dict)
