"""Minimal file-based registry for SKUs, trained agents and evaluations."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping, Optional

from control_rl.agent_specs import normalize_algorithm_name
from model.eval_storage import append_evaluation_index, save_evaluation_artifacts
from model.io import now_utc_iso, read_json, slugify, write_json


class SKURegistry:
    def __init__(self, root_dir: str | Path = "artifacts/sku_registry") -> None:
        self.root = Path(root_dir)
        self.skus_dir = self.root / "skus"
        self.skus_dir.mkdir(parents=True, exist_ok=True)

    def _sku_dir(self, sku_id: str) -> Path:
        return self.skus_dir / slugify(sku_id)

    def _sku_path(self, sku_id: str) -> Path:
        return self._sku_dir(sku_id) / "sku.json"

    def _agents_dir(self, sku_id: str) -> Path:
        return self._sku_dir(sku_id) / "agents"

    def _agent_dir(self, sku_id: str, agent_id: str) -> Path:
        return self._agents_dir(sku_id) / slugify(agent_id)

    def _agent_path(self, sku_id: str, agent_id: str) -> Path:
        return self._agent_dir(sku_id, agent_id) / "agent.json"

    def ensure_sku(
        self,
        sku_id: str,
        sku_config: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if not str(sku_id).strip():
            raise ValueError("sku_id must be non-empty")

        path = self._sku_path(sku_id)
        if path.exists() and not overwrite:
            return read_json(path)

        payload = {
            "sku_id": str(sku_id),
            "created_at": now_utc_iso(),
            "sku_config": dict(sku_config),
        }
        write_json(path, payload)
        self._agents_dir(sku_id).mkdir(parents=True, exist_ok=True)
        return payload

    def list_skus(self) -> list[dict[str, Any]]:
        return [read_json(path) for path in sorted(self.skus_dir.glob("*/sku.json"))]

    def get_sku(self, sku_id: str) -> dict[str, Any]:
        path = self._sku_path(sku_id)
        if not path.exists():
            raise FileNotFoundError(f"SKU '{sku_id}' not found")
        return read_json(path)

    def get_sku_config(self, sku_id: str) -> dict[str, Any]:
        sku = self.get_sku(sku_id)
        return dict(sku["sku_config"])

    def get_agent(self, sku_id: str, agent_id: str) -> dict[str, Any]:
        path = self._agent_path(sku_id, agent_id)
        if not path.exists():
            raise FileNotFoundError(f"Agent '{agent_id}' not found for SKU '{sku_id}'")
        return read_json(path)

    def find_agent(
        self,
        sku_id: str,
        algorithm_name: str,
        algorithm_config_key: str,
    ) -> dict[str, Any]:
        algo = normalize_algorithm_name(algorithm_name).lower()
        agent_id = f"{algo}_{str(algorithm_config_key).strip()}"
        return self.get_agent(sku_id, agent_id)

    def agent_paths(self, sku_id: str, agent_id: str) -> dict[str, str]:
        agent_dir = self._agent_dir(sku_id, agent_id)
        eval_dir = agent_dir / "evaluations"
        return {
            "agent_dir": str(agent_dir.resolve()),
            "agent_path": str((agent_dir / "agent.json").resolve()),
            "evaluation_dir": str(eval_dir.resolve()),
            "evaluation_index": str((eval_dir / "evaluations.json").resolve()),
        }

    def list_agents(self, sku_id: str) -> list[dict[str, Any]]:
        agents_dir = self._agents_dir(sku_id)
        if not agents_dir.exists():
            return []
        return [read_json(path) for path in sorted(agents_dir.glob("*/agent.json"))]

    def save_agent(self, agent_metadata: Mapping[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
        required = (
            "agent_id",
            "sku_id",
            "algorithm_name",
            "algorithm_config",
            "algorithm_config_key",
            "training_config",
            "env_config",
            "model_path",
            "status_path",
            "learning_curves_dir",
            "action_mode",
            "created_at",
            "evaluation_dir",
            "evaluation_index",
        )
        missing = [name for name in required if name not in agent_metadata]
        if missing:
            raise ValueError(f"agent_metadata is missing required fields: {missing}")

        sku_id = str(agent_metadata["sku_id"])
        agent_id = str(agent_metadata["agent_id"])
        self.ensure_sku(sku_id, dict(agent_metadata["env_config"]))

        path = self._agent_path(sku_id, agent_id)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Agent '{agent_id}' already exists for SKU '{sku_id}'")

        payload = dict(agent_metadata)
        write_json(path, payload)
        return payload

    def delete_agent(
        self,
        sku_id: str,
        agent_id: str,
        *,
        delete_artifacts: bool = False,
    ) -> dict[str, Any]:
        agent = self.get_agent(sku_id, agent_id)

        if delete_artifacts:
            raw_run_dir = str(Path(str(agent["model_path"])).expanduser().resolve().parent).strip()
            if raw_run_dir:
                run_dir = Path(raw_run_dir).expanduser()
                shutil.rmtree(run_dir, ignore_errors=True)

        agent_dir = self._agent_dir(sku_id, agent_id)
        shutil.rmtree(agent_dir, ignore_errors=True)
        return agent

    def delete_sku(
        self,
        sku_id: str,
        *,
        delete_artifacts: bool = False,
    ) -> dict[str, Any]:
        sku = self.get_sku(sku_id)
        deleted_agents = []
        for agent in self.list_agents(sku_id):
            deleted_agents.append(
                self.delete_agent(
                    sku_id,
                    str(agent["agent_id"]),
                    delete_artifacts=delete_artifacts,
                )
            )

        shutil.rmtree(self._sku_dir(sku_id), ignore_errors=True)
        return {
            "sku": sku,
            "agents": deleted_agents,
        }

    def save_evaluation(
        self,
        *,
        sku_id: str,
        agent_id: str,
        raw_steps,
        kpi_summary,
        metadata: Mapping[str, Any],
        evaluation_id: Optional[str] = None,
    ) -> dict[str, Any]:
        agent = self.get_agent(sku_id, agent_id)
        eval_dir = Path(agent["evaluation_dir"])
        eval_index = Path(agent["evaluation_index"])

        saved = save_evaluation_artifacts(
            output_dir=eval_dir,
            raw_steps=raw_steps,
            kpi_summary=kpi_summary,
            metadata={
                "sku_id": sku_id,
                "agent_id": agent_id,
                **dict(metadata),
            },
            evaluation_id=evaluation_id,
        )
        index_path = append_evaluation_index(
            index_path=eval_index,
            evaluation_id=str(saved["evaluation_id"]),
            created_at=str(saved["created_at"]),
            metadata_path=str(saved["metadata_path"]),
        )
        return {
            "evaluation_id": str(saved["evaluation_id"]),
            "evaluation_dir": str(saved["evaluation_dir"]),
            "metadata_path": str(saved["metadata_path"]),
            "raw_steps_path": str(saved["raw_steps_path"]),
            "kpi_summary_path": str(saved["kpi_summary_path"]),
            "index_path": str(index_path),
        }
