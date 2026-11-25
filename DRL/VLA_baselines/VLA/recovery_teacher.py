"""Privileged DEMONSTRATION teacher with retry; never a learned-policy fallback."""
from VLA.common.scripted_policy import ScriptedPolicy


class RecoveryTeacher(ScriptedPolicy):
    def reset(self, task):
        super().reset(task)
        self.recovery_events = []

    def predict(self, observation):
        state = self.env.debug_state()
        lost = self.phase in ("lift", "transport", "lower", "hold") and not state["grasped"]
        timed_out = self.phase == "close" and self.phase_steps >= 15 and not state["grasped"]
        if lost or timed_out:
            self.recovery_events.append({"step": int(observation.step), "from_phase": self.phase,
                                         "reason": "lost_grasp" if lost else "close_without_grasp"})
            self._advance("approach")
        # approach emits OPEN before returning to the object; lifting is never
        # entered via the legacy close-timeout branch while the hand is empty.
        return super().predict(observation)
