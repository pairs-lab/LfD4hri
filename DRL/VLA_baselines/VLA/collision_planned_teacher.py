"""Demonstration-only midbody/yaw teacher with verified-grasp retry."""
import numpy as np
import pybullet as p

from VLA.collision_grasp_planner import plan_body_grasp, wrap_axis_angle
from VLA.grasp_contact_prototype import MidbodyContactTeacher


class CollisionPlannedTeacher(MidbodyContactTeacher):
    def reset(self, task):
        super().reset(task)
        plan = plan_body_grasp(self.env, self.env.pick_id)
        position, quaternion = p.getBasePositionAndOrientation(self.env.pick_id)
        rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        self.point_local = rotation.T @ (plan["point_world"] - position)
        self.axis_local = None if plan["jaw_axis_world"] is None else rotation.T @ np.r_[plan["jaw_axis_world"], 0]
        self.plan = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in plan.items()}
        self.carry_stage = None
        surface = plan.get("touch_surface_world")
        if surface is None:
            lo, hi = map(np.asarray, p.getAABB(self.env.pick_id))
            surface = np.r_[(lo[:2]+hi[:2])/2, hi[2]]
        lo, hi = map(np.asarray, p.getAABB(self.env.pick_id))
        ray_from = np.r_[surface[:2], hi[2]+.015]
        ray_to = np.r_[surface[:2], lo[2]-.005]
        hit = p.rayTest(ray_from.tolist(), ray_to.tolist())[0]
        if hit[0] == self.env.pick_id:
            # Mesh quantile height can be above the real surface at this XY.
            # Teacher-only raycast selects the actual collision surface.
            surface = np.asarray(hit[3]) - [0, 0, .001]
        self.touch_surface_local = rotation.T @ (surface - position)

    def target_and_rotation(self):
        position, quaternion = p.getBasePositionAndOrientation(self.env.pick_id)
        rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        target = rotation @ self.point_local + position
        target[2] = max(float(target[2]), p.getAABB(self.env.table_id)[1][2] + .0205)
        yaw_delta = 0.
        if self.axis_local is not None:
            desired = rotation @ self.axis_local
            left, right = self.env.gripper.pad_positions()
            current = right - left
            error = wrap_axis_angle(np.arctan2(desired[1], desired[0]) - np.arctan2(current[1], current[0]))
            pose = p.getLinkState(self.env.robot_id, self.env.robot_spec.mount_link, computeForwardKinematics=True)
            tool = np.asarray(p.getMatrixFromQuaternion(pose[5])).reshape(3, 3)
            if abs(tool[2, 2]) < .9:
                raise ValueError("Planar yaw controller requires top-down tool orientation")
            yaw_delta = float(np.clip(error / tool[2, 2], -.12, .12))
        return target, yaw_delta

    def predict(self, observation):
        if self.task.name == "touch":
            if self.env._last_success:
                # Teacher-only collection can require a success streak. Once
                # contact/proximity is achieved, HOLD rather than continue
                # pushing through the object for the remaining buffer ticks.
                # Evaluator feedback is never supplied to learned policies.
                return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
            position, quaternion = p.getBasePositionAndOrientation(self.env.pick_id)
            rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
            surface = rotation @ self.touch_surface_local + position
            centre = self.env.gripper.grasp_center()
            pads = self.env.gripper.world_pad_vertices()
            targets = []
            for vertices in pads:
                low = vertices[:, 2].min()
                tip = vertices[vertices[:, 2] <= low + .001].mean(axis=0)
                target = surface - (tip-centre)
                if -.36 <= target[0] <= .36 and -.79 <= target[1] <= -.21:
                    targets.append(target)
            if not targets:
                raise ValueError("No safely reachable physical touch point")
            target = min(targets, key=lambda point: np.linalg.norm(point[:2]-centre[:2])).copy()
            if self.phase == "approach":
                target[2] = max(float(target[2])+.14,
                    max(p.getAABB(body)[1][2] for body in self.env.object_ids.values())+.04)
                if self._near(centre, target, .015):
                    self._advance("descend")
                return self._move(centre, target, -1)
            # Fine, slow contact, rather than forcing a finger through a hard
            # object for the rest of a 100ms control interval.
            return np.r_[np.clip(.35*(target-centre), -.004, .004), np.zeros(3), -1]
        if self.phase in ("approach", "descend", "close"):
            state = self.env.debug_state()
            if self.phase == "close" and not state["grasped"] and self.phase_steps >= 15:
                return super().predict(observation)
            target, yaw = self.target_and_rotation()
            if self.phase == "approach":
                target[2] += .14
                target[2] = max(float(target[2]), max(p.getAABB(body)[1][2] for body in self.env.object_ids.values()) + .04)
                if self._near(state["grasp_center"], target, .015) and abs(yaw) < .025:
                    self._advance("descend")
                command = -1
            elif self.phase == "descend":
                command = -1
                if self._near(state["grasp_center"], target, .004) and abs(yaw) < .025:
                    self._advance("close")
            else:
                command = 1
                if state["grasped"]:
                    self._advance("lift")
            self.phase_steps += 1
            action = self._move(state["grasp_center"], target, command)
            action[5] = yaw
            return action
        if self.phase == "transport" and self.task.name in ("move", "pick_place"):
            state = self.env.debug_state()
            if not state["grasped"]:
                self.carry_stage = None
                return super().predict(observation)
            obj = state["pick_position"]
            if self.carry_stage is None:
                top = max(p.getAABB(body)[1][2] for body in self.env.object_ids.values())
                self.carry_height = float(obj[2] + max(0., top + .04 - state["pick_bottom"][2]))
                self.carry_stage = "raise"
            target = obj.copy() if self.carry_stage == "raise" else state["goal_position"].copy()
            if self.carry_stage != "lower":
                target[2] = self.carry_height
            if self._near(obj, target, .012):
                self.carry_stage = "travel" if self.carry_stage == "raise" else "lower"
            self.phase_steps += 1
            return self._move(state["grasp_center"], state["grasp_center"] + target - obj, 1)
        return super().predict(observation)
