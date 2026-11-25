#!/usr/bin/env python3
"""Serve official pi0/pi0-FAST checkpoints with the UR5e data contract.

The upstream UR5 document is an outline rather than an executable server.  This
file supplies the missing transform/config glue without changing openpi itself.
Run it from the isolated openpi environment; ``VLA.pi0.adapter`` is the client.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
from pathlib import Path

import numpy as np


class _PolicyWithMetadata:
    """Attach a verified runtime contract without changing openpi internals."""

    def __init__(self, policy, metadata: dict) -> None:
        self._policy = policy
        self._metadata = metadata

    @property
    def metadata(self) -> dict:
        return self._metadata

    def infer(self, observation: dict, **kwargs) -> dict:
        return self._policy.infer(observation, **kwargs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_checkpoint_metadata(
    checkpoint: str,
    *,
    model_name: str,
    repo_id: str,
    train_config,
    lineage: dict,
) -> dict:
    """Verify weights/config/stats identity before exposing a target policy."""

    from VLA.fingerprint_checkpoint import REPORT_NAME, layout_records, layout_sha256
    from VLA.pi0.training import CHECKPOINT_MANIFEST_NAME

    root = Path(checkpoint).resolve()
    contract_path = root / CHECKPOINT_MANIFEST_NAME
    fingerprint_path = root / REPORT_NAME
    for path in (contract_path, fingerprint_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Target checkpoint is not finalized: missing {path}. "
                "Create its VLA manifest and full fingerprint before serving it."
            )
    contract = json.loads(contract_path.read_text())
    expected_slots = (
        ["base_0_rgb", "left_wrist_0_rgb"]
        if model_name == "pi0"
        else ["base_0_rgb", "base_1_rgb_zero_padding", "wrist_0_rgb"]
    )
    asset_id = train_config.data.assets.asset_id or repo_id
    expected_model_config = {
        key: getattr(train_config.model, key, None)
        for key in ("paligemma_variant", "action_expert_variant", "max_token_len", "fast_model_tokenizer_kwargs")
    }
    if (
        contract.get("version") != 1
        or Path(contract.get("checkpoint", "")).resolve() != root
        or contract.get("model") != model_name
        or contract.get("model_type") != str(train_config.model.model_type)
        or contract.get("action_horizon") != int(train_config.model.action_horizon)
        or contract.get("action_dim") != int(train_config.model.action_dim)
        or contract.get("camera_slots") != expected_slots
        or contract.get("repo_id") != repo_id
        or contract.get("asset_id", repo_id) != asset_id
        or contract.get("model_config") != expected_model_config
        or contract.get("control_hz") != 10
        or contract.get("source", {}).get("index_sha256") != lineage["index_sha256"]
        or contract.get("source", {}).get("manifest_sha256")
        != lineage["manifest_sha256"]
    ):
        raise RuntimeError("Checkpoint VLA manifest does not match the requested runtime config")

    if model_name == "pi0_fast" and contract.get("interpretation_code", {}).get(
        "fast_tokenizer_sha256"
    ) != _sha256(Path(__file__).resolve().parent / "fast_tokenizer.py"):
        raise RuntimeError("FAST tokenizer changed after checkpoint finalization")
    local_stats = train_config.assets_dirs / asset_id / "norm_stats.json"
    checkpoint_stats = root / "assets" / asset_id / "norm_stats.json"
    expected_stats_hash = contract.get("normalization", {}).get("norm_stats_sha256")
    if (
        not local_stats.is_file()
        or not checkpoint_stats.is_file()
        or expected_stats_hash != _sha256(local_stats)
        or expected_stats_hash != _sha256(checkpoint_stats)
    ):
        raise RuntimeError("Checkpoint normalization statistics are missing or mismatched")

    from VLA.fingerprint_checkpoint import verify_fingerprint

    fingerprint = verify_fingerprint(root)
    if (
        fingerprint.get("version") != 1
        or Path(fingerprint.get("checkpoint", "")).resolve() != root
        or fingerprint.get("layout_sha256") != layout_sha256(layout_records(root))
        or not fingerprint.get("content_tree_sha256")
    ):
        raise RuntimeError("Checkpoint content fingerprint is missing or stale")
    return {
        "protocol": "pybullet_ur5e_robotiq_openpi_v1",
        "target_domain": True,
        "model": model_name,
        "repo_id": repo_id,
        "asset_id": asset_id,
        "action_horizon": int(train_config.model.action_horizon),
        "action_dim": int(train_config.model.action_dim),
        "control_hz": 10,
        "checkpoint": {
            "path": str(root),
            "content_tree_sha256": fingerprint["content_tree_sha256"],
            "layout_sha256": fingerprint["layout_sha256"],
            "fingerprint_report_sha256": _sha256(fingerprint_path),
            "contract_sha256": _sha256(contract_path),
            "norm_stats_sha256": expected_stats_hash,
        },
    }


class _RequestNoisePolicy:
    """Pass optional client-supplied diffusion noise outside model inputs."""

    def __init__(self, policy) -> None:
        self._policy = policy

    @property
    def metadata(self):
        return self._policy.metadata

    def infer(self, observation: dict) -> dict:
        observation = dict(observation)
        noise = observation.pop("__policy_noise__", None)
        return self._policy.infer(observation, noise=noise)


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def build_policy(
    model_name: str,
    checkpoint: str,
    repo_id: str | None = None,
    *,
    lerobot_home: Path | None = None,
    source_dataset: Path | None = None,
    audit_report: Path | None = None,
    roundtrip_report: Path | None = None,
    action_adapter_report: Path | None = None,
    closed_loop_adapter_report: Path | None = None,
    action_horizon: int | None = None,
    asset_id: str | None = None,
    fast_codec_report: Path | None = None,
):
    from openpi import transforms
    from openpi.models import model as model_api
    from openpi.models import pi0_config
    from openpi.models import pi0_fast
    from openpi.policies import policy_config
    from openpi.training import config
    from VLA.pi0.fast_tokenizer import UR5FASTTokenizer

    @dataclasses.dataclass(frozen=True)
    class UR5Inputs(transforms.DataTransformFn):
        model_type: model_api.ModelType

        def __call__(self, data: dict) -> dict:
            state = np.concatenate(
                (
                    np.asarray(data["joints"], dtype=np.float32).reshape(6),
                    np.asarray(data["gripper"], dtype=np.float32).reshape(1),
                )
            )
            # PI0-FAST tokenizes the normalized state directly and its base
            # checkpoint/norm statistics use the model's full 32D action
            # space. PI0 has a separate PadStatesAndActions model transform,
            # while FAST does not, so pad the UR5's 7D state here.
            if self.model_type == model_api.ModelType.PI0_FAST:
                state = np.pad(state, (0, 32 - state.shape[0]))
            base = _parse_image(data["base_rgb"])
            wrist = _parse_image(data["wrist_rgb"])
            if self.model_type == model_api.ModelType.PI0:
                names = ("base_0_rgb", "left_wrist_0_rgb")
                images = (base, wrist)
                masks = (np.True_, np.True_)
            elif self.model_type == model_api.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base, np.zeros_like(base), wrist)
                # FAST tokenization cannot mask a missing view; match upstream
                # DroidInputs and fill the second base camera with zeros.
                masks = (np.True_, np.True_, np.True_)
            else:
                raise ValueError(f"Unsupported model type {self.model_type}")
            result = {
                "state": state,
                "image": dict(zip(names, images, strict=True)),
                "image_mask": dict(zip(names, masks, strict=True)),
            }
            if "actions" in data:
                result["actions"] = np.asarray(data["actions"])
            if "prompt" in data:
                result["prompt"] = data["prompt"]
            return result

    @dataclasses.dataclass(frozen=True)
    class UR5Outputs(transforms.DataTransformFn):
        def __call__(self, data: dict) -> dict:
            return {"actions": np.asarray(data["actions"])[..., :7]}

    # A local target-domain checkpoint must be restored with the exact model
    # shape, camera transforms, native UR5 action representation and dataset
    # asset id used during training.  Reconstructing a base-model config here
    # silently loads the wrong normalization (and for FAST, the wrong 32D
    # output shape), so use the single training config as the source of truth.
    if repo_id is not None:
        from types import SimpleNamespace

        from VLA.pi0.training import (
            CHECKPOINT_MANIFEST_NAME,
            DEFAULT_ACTION_ADAPTER_AUDIT,
            DEFAULT_AUDIT,
            DEFAULT_CLOSED_LOOP_ADAPTER_AUDIT,
            DEFAULT_LEROBOT_HOME,
            DEFAULT_ROUNDTRIP_AUDIT,
            DEFAULT_SOURCE,
            make_config,
            validate_lineage,
        )

        contract_path = Path(checkpoint).resolve() / CHECKPOINT_MANIFEST_NAME
        if not contract_path.is_file():
            raise FileNotFoundError(f"Finalize the target checkpoint manifest first: {contract_path}")
        checkpoint_contract = json.loads(contract_path.read_text())
        resolved_horizon = (
            int(checkpoint_contract["action_horizon"])
            if action_horizon is None
            else action_horizon
        )
        resolved_asset_id = asset_id or checkpoint_contract.get("asset_id", repo_id)
        codec_path = checkpoint_contract.get("audit_reports", {}).get("fast_codec", {}).get("path")
        tokenizer_config = checkpoint_contract.get("model_config", {}).get("fast_model_tokenizer_kwargs") or {}

        config_args = SimpleNamespace(
            model=model_name,
            action_horizon=resolved_horizon,
            asset_id=resolved_asset_id,
            fast_codec_report=fast_codec_report or (Path(codec_path) if codec_path else None),
            fast_codec_scale=float(tokenizer_config.get("codec_scale", 10.0)),
            fast_tokenizer_path=tokenizer_config.get("fast_tokenizer_path", "physical-intelligence/fast"),
            repo_id=repo_id,
            lerobot_home=(lerobot_home or DEFAULT_LEROBOT_HOME).resolve(),
            source_dataset=(source_dataset or DEFAULT_SOURCE).resolve(),
            audit_report=(audit_report or DEFAULT_AUDIT).resolve(),
            roundtrip_report=(roundtrip_report or DEFAULT_ROUNDTRIP_AUDIT).resolve(),
            action_adapter_report=(
                action_adapter_report or DEFAULT_ACTION_ADAPTER_AUDIT
            ).resolve(),
            closed_loop_adapter_report=(
                closed_loop_adapter_report or DEFAULT_CLOSED_LOOP_ADAPTER_AUDIT
            ).resolve(),
            steps=1,
            batch_size=1,
            save_interval=1000,
            exp_name="inference",
            max_frames=None,
            overwrite=False,
        )
        train_config = make_config(config_args)
        lineage = validate_lineage(config_args, train_config, require_stats=True)
        metadata = _verified_checkpoint_metadata(
            checkpoint,
            model_name=model_name,
            repo_id=repo_id,
            train_config=train_config,
            lineage=lineage,
        )
        policy = policy_config.create_trained_policy(train_config, checkpoint)
        return _PolicyWithMetadata(policy, metadata)

    if model_name == "pi0":
        model_config = pi0_config.Pi0Config()
    elif model_name == "pi0_fast":
        model_config = pi0_fast.Pi0FASTConfig(fast_model_tokenizer=UR5FASTTokenizer)
    else:
        raise ValueError("model must be pi0 or pi0_fast")

    def ur5_transforms(current_model):
        group = transforms.Group(
            inputs=[UR5Inputs(model_type=current_model.model_type)],
            outputs=[UR5Outputs()],
        )
        # The published UR5 recipe models six absolute joint targets as deltas
        # during training and restores absolute targets during inference.
        delta_mask = transforms.make_bool_mask(6, -1)
        return group.push(
            inputs=[transforms.DeltaActions(delta_mask)],
            outputs=[transforms.AbsoluteActions(delta_mask)],
        )

    train_config = config.TrainConfig(
        name=f"{model_name}_pybullet_ur5e",
        model=model_config,
        data=config.SimpleDataConfig(
            assets=config.AssetsConfig(
                assets_dir=f"{checkpoint}/assets",
                asset_id="ur5e",
            ),
            data_transforms=ur5_transforms,
            base_config=config.DataConfig(prompt_from_task=True),
        ),
    )
    policy = policy_config.create_trained_policy(train_config, checkpoint)
    return _PolicyWithMetadata(
        policy,
        {
            "protocol": "pybullet_ur5e_robotiq_openpi_v1",
            "target_domain": False,
            "model": model_name,
            "repo_id": None,
            "action_horizon": int(model_config.action_horizon),
            "action_dim": int(model_config.action_dim),
            "checkpoint": {"path": checkpoint},
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("pi0", "pi0_fast"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Training dataset id for a target-domain checkpoint; omit only for official base checkpoints",
    )
    parser.add_argument("--lerobot-home", type=Path, default=None)
    parser.add_argument("--source-dataset", type=Path, default=None)
    parser.add_argument("--audit-report", type=Path, default=None)
    parser.add_argument("--roundtrip-report", type=Path, default=None)
    parser.add_argument("--action-adapter-report", type=Path, default=None)
    parser.add_argument("--closed-loop-adapter-report", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action-horizon", type=int, choices=(5, 10), default=None)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--fast-codec-report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    checkpoint = args.checkpoint or f"gs://openpi-assets/checkpoints/{args.model}_base"
    lineage_paths = (
        args.source_dataset,
        args.audit_report,
        args.roundtrip_report,
        args.action_adapter_report,
        args.closed_loop_adapter_report,
    )
    if args.repo_id is None and any(path is not None for path in lineage_paths):
        raise ValueError("Lineage reports apply only to a target checkpoint with --repo-id")
    policy = build_policy(
        args.model,
        checkpoint,
        repo_id=args.repo_id,
        lerobot_home=args.lerobot_home,
        source_dataset=args.source_dataset,
        audit_report=args.audit_report,
        roundtrip_report=args.roundtrip_report,
        action_adapter_report=args.action_adapter_report,
        closed_loop_adapter_report=args.closed_loop_adapter_report,
        action_horizon=args.action_horizon,
        asset_id=args.asset_id,
        fast_codec_report=args.fast_codec_report,
    )
    if args.model == "pi0":
        policy = _RequestNoisePolicy(policy)
    WebsocketPolicyServer(
        policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
