#!/usr/bin/env python3
"""
读取 moe_config.yaml 并将指定 preset 的参数打印为 train.py 命令行参数。

用法：
  python3 scripts/moe_args.py                        # 使用 active_preset
  python3 scripts/moe_args.py --preset minimal       # 指定 preset
  python3 scripts/moe_args.py --preset off           # 禁用 MoE

在 shell 脚本里：
  MOE_ARGS=$(python3 scripts/moe_args.py --preset standard)
  torchrun ... \\
      --other_arg value \\
      $MOE_ARGS
"""

import argparse
import os
import sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML not found. Run: pip install pyyaml")


def main():
    parser = argparse.ArgumentParser(description="Convert moe_config.yaml preset to CLI args.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "moe_config.yaml"),
        help="Path to moe_config.yaml (default: scripts/moe_config.yaml)",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Preset name to use (default: active_preset in the YAML file)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available preset names and exit",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.list:
        print("Available presets:", ", ".join(cfg["presets"].keys()))
        print("Current active_preset:", cfg.get("active_preset", "(not set)"))
        return

    preset_name = args.preset or cfg.get("active_preset")
    if preset_name not in cfg["presets"]:
        sys.exit(
            f"Preset '{preset_name}' not found. "
            f"Available: {list(cfg['presets'].keys())}"
        )

    preset = cfg["presets"][preset_name]
    parts = []
    # Fixed-forward diversity warmup was replaced by the GA-aware ratio in
    # train.py. Keep old YAML files readable, but do not emit their stale key.
    ignored_legacy_keys = {"moe_diversity_warmup_steps"}
    for key, value in preset.items():
        if key in ignored_legacy_keys:
            continue
        if isinstance(value, bool):
            # argparse expects "True"/"False" strings
            parts.append(f"--{key} {value}")
        else:
            parts.append(f"--{key} {value}")

    print(" ".join(parts))


if __name__ == "__main__":
    main()
