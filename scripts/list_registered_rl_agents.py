#!/usr/bin/env python
"""List registered SKUs and their persisted RL agents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control_rl.registry import SKURegistry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List registered SKUs and RL agents")
    parser.add_argument(
        "--registry-root",
        default="artifacts/sku_registry",
        help="Path to the SKU registry root",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the summary as JSON instead of human-readable text",
    )
    return parser.parse_args()


def _build_summary(registry: SKURegistry) -> list[dict]:
    summary: list[dict] = []
    for sku in registry.list_skus():
        sku_id = str(sku["sku_id"])
        agents = registry.list_agents(sku_id)
        summary.append(
            {
                "sku_id": sku_id,
                "created_at": sku.get("created_at"),
                "sku_config": sku.get("sku_config", {}),
                "agents": [
                    {
                        "agent_id": agent["agent_id"],
                        "algorithm_name": agent["algorithm_name"],
                        "algorithm_config_key": agent["algorithm_config_key"],
                        "action_mode": agent["action_mode"],
                        "algorithm_config": agent["algorithm_config"],
                        "training_config": agent["training_config"],
                        "model_path": agent["model_path"],
                    }
                    for agent in agents
                ],
            }
        )
    return summary


def _print_human(summary: list[dict]) -> None:
    if not summary:
        print("No registered SKUs found.")
        return

    for sku in summary:
        print(f"SKU: {sku['sku_id']}")
        print(f"  Created at: {sku.get('created_at')}")
        print(f"  SKU config: {json.dumps(sku.get('sku_config', {}), sort_keys=True)}")
        print(f"  Agents: {len(sku['agents'])}")
        if not sku["agents"]:
            print("  No trained agents registered.")
            continue

        for agent in sku["agents"]:
            print(f"  - Agent: {agent['agent_id']}")
            print(f"    Algorithm: {agent['algorithm_name']}")
            print(f"    Config key: {agent['algorithm_config_key']}")
            print(f"    Action mode: {agent['action_mode']}")
            print(f"    Algorithm config: {json.dumps(agent['algorithm_config'], sort_keys=True)}")
            print(f"    Training config: {json.dumps(agent['training_config'], sort_keys=True)}")
            print(f"    Model path: {agent['model_path']}")


def main() -> int:
    args = _parse_args()
    registry = SKURegistry(args.registry_root)
    summary = _build_summary(registry)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_human(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
