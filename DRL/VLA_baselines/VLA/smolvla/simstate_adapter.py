"""Explicit privileged VLA variant: neural actions ONLY, no waypoint/oracle fallback."""
import json
from pathlib import Path

from VLA.common.provenance import sha256_file
from VLA.simstate_feedback import SCHEMA, FEATURE_NAMES, feedback_state
from VLA.smolvla.adapter import SmolVLAPolicyAdapter


class SimStateSmolVLA(SmolVLAPolicyAdapter):
    name = "smolvla_simstate"
    comparable = False  # separate privileged-input comparison row, never plain VLA
    input_modalities = SmolVLAPolicyAdapter.input_modalities + (
        "pad_kinematic_position", "previous_gripper_command", "PRIVILEGED_requested_object_relative_pose",
        "PRIVILEGED_goal_object_relative_pose", "PRIVILEGED_verified_correct_body_grasp")

    def __init__(self, env, checkpoint, **kwargs):
        contract = json.loads((Path(checkpoint) / "simstate_contract.json").read_text())
        if (contract.get("schema") != SCHEMA or contract.get("feature_names") != FEATURE_NAMES
                or not contract.get("privileged_model_input") or contract.get("oracle_controller_at_deployment") is not False
                or contract.get("feature_builder_sha256") != sha256_file(Path(__file__).resolve().parents[1] / "simstate_feedback.py")
                or json.loads((Path(checkpoint) / "config.json").read_text())["input_features"]["observation.state"]["shape"] != [18]):
            raise ValueError("Privileged checkpoint input/normalization contract missing or changed")
        super().__init__(env, checkpoint=checkpoint, **kwargs)

    def build_sensor_batch(self, observation):
        import torch
        batch = super().build_sensor_batch(observation)
        batch["observation.state"] = torch.from_numpy(feedback_state(self.env, observation)).unsqueeze(0)
        return batch
