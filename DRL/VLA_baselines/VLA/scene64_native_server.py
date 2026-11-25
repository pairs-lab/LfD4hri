"""All64 causal simulator fields into the REAL native VLA language path.

Process-local compatibility bridge; no frozen18 runtime file is edited.
Native architecture, sampler, weights and action convention are retained.
1024-token budget and unavailable-camera masking are explicit NEW contracts.
This module does not claim all four families have been fine-tuned.
"""
import json
from dataclasses import dataclass,replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from VLA.scene64_conditioning import PROMPT_SCHEMA,augment_prompt,parse_prompt,validate_scene
from VLA.common.artifact_io import atomic_json


def main():
    import flask
    import VLA.privileged_native_server as native
    original_config=native.openpi_config
    def config(model_name,checkpoint):
        cfg=original_config(model_name,checkpoint)
        if model_name!="pi0_fast":return cfg
        from openpi import transforms
        @dataclass(frozen=True)
        class MaskUnavailable(transforms.DataTransformFn):
            def __call__(self,data):
                result=dict(data);result["image_mask"]={**data["image_mask"],"base_1_rgb":np.False_};return result
        stock=cfg.data.data_transforms
        def groups(model):
            group=stock(model)
            return transforms.Group(inputs=[group.inputs[0],MaskUnavailable(),*group.inputs[1:]],outputs=group.outputs)
        return replace(cfg,data=replace(cfg.data,data_transforms=groups))
    native.openpi_config=config;native.TOKEN_LIMIT=1024
    native.PROMPT_SCHEMA=PROMPT_SCHEMA;native.augment_prompt=augment_prompt
    native.parse_feedback=parse_prompt;native.validate_state=validate_scene
    OriginalFlask=flask.Flask
    class SceneFlask(OriginalFlask):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            @self.before_request
            def accept():
                if flask.request.path=="/act":
                    payload=flask.request.get_json()
                    if payload.get("schema")!=PROMPT_SCHEMA or "scene64" not in payload:
                        return flask.jsonify(error="Explicit full64 schema/payload required"),409
                    # Legacy internal variable name is not the wire contract.
                    # Its validator is replaced by the full64 fail-closed check.
                    payload["state18"]=payload["scene64"]
            @self.after_request
            def metadata(response):
                if flask.request.path=="/metadata" and response.status_code==200:
                    info=response.get_json();info.update(actual_feedback_fields=64,
                        unavailable_camera_masked=info["model"]=="pi0_fast",
                        input_only_bridge_no_waypoint_or_action_controller=True)
                    response.set_data(json.dumps(info));response.content_type="application/json"
                return response
    with patch.object(flask,"Flask",SceneFlask):native.main()


if __name__ == "__main__":main()
