"""Explicit runtime dispatch prevents V9 data from using the legacy gripper."""
from VLA.common.environment import ManipulationEnv
from VLA.common.observation_profile import ProfiledManipulationEnv


def environment_class_for_profile(profile):
    if profile is None:
        return ManipulationEnv
    contract = profile.get("gripper_controller")
    if contract is None:
        return ProfiledManipulationEnv
    if contract == "body_contact_v9":
        from VLA.common.body_environment import BodyGraspEnvironment
        return BodyGraspEnvironment
    raise ValueError(f"Unknown gripper-controller contract: {contract}")
