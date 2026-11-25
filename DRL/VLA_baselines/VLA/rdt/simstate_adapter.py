"""Explicit privileged RDT client; no controller or oracle action fallback."""
from pathlib import Path

from VLA.common.provenance import sha256_file
from VLA.rdt.adapter import RDTPolicy
from VLA.rdt.simstate_inputs import SCHEMA
from VLA.simstate_feedback import feedback_state


class SimStateRDTPolicy(RDTPolicy):
    name = "rdt_simstate"
    comparable = False
    input_modalities = RDTPolicy.input_modalities + ("PRIVILEGED_causal_target_goal_feedback11",)

    def _verify_server(self):
        super()._verify_server()
        metadata = self.server_metadata
        if (metadata.get("simstate_schema") != SCHEMA or metadata.get("privileged_model_inputs") is not True
                or metadata.get("oracle_controller_at_deployment") is not False
                or metadata.get("feature_builder_sha256") != sha256_file(Path(__file__).resolve().parents[1]/"simstate_feedback.py")):
            raise ValueError("A separately qualified privileged server is required; never route STATE18 to an ordinary model")

    def build_sensor_request(self, observation, previous):
        payload = super().build_sensor_request(observation, previous)
        payload.update(simstate_schema=SCHEMA, privileged_state18=feedback_state(self.env, observation).tolist())
        return payload
