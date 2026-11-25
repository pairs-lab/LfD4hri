"""Offline teacher-only, collision-free near-object missed-grasp starts.

These controls deliberately close beside (not around) a physical scene object.
They are perturbations, never expert labels or a learned-policy/runtime fallback.
No object is repositioned, no target-specific controller or guard is modified.
"""
import numpy as np
import pybullet as p


MIN_OBJECT_CLEARANCE_M = .003
MAX_NEAR_OBJECT_DISTANCE_M = .11
MIN_CLOSED_FK = .95


def _gripper_aabb(env):
    bounds = [p.getAABB(env.gripper.body, link)
              for link in range(-1, p.getNumJoints(env.gripper.body))]
    return np.min([b[0] for b in bounds], axis=0), np.max([b[1] for b in bounds], axis=0)


def _distance_to_box(point, low, high):
    return float(np.linalg.norm(np.maximum(np.maximum(low-point, point-high), 0)))


def near_object_candidates(bounds, offsets, table_top):
    """All-object geometry only: no selected target, primitive or text input.

    Conservative OPEN-gripper XY footprint separates the entire gripper from
    object AABBs before the attempted close. Actual collision distances and
    contacts are independently checked while executing the bounded approach.
    """
    lower, upper = map(np.asarray, offsets)
    result = []
    for name, low, high in sorted(bounds, key=lambda item: item[0]):
        low, high = np.asarray(low), np.asarray(high)
        centre = (low+high)/2
        height = max(float(centre[2]), float(table_top-lower[2]+.010))
        if height > high[2]+.020:
            continue  # This cannot honestly be a near-body miss at this pose.
        margin = .010
        # Near each cardinal face; all layouts get the same candidate rule.
        xys = ((low[0]-upper[0]-margin, centre[1]),
               (high[0]-lower[0]+margin, centre[1]),
               (centre[0], low[1]-upper[1]-margin),
               (centre[0], high[1]-lower[1]+margin))
        for xy in xys:
            target = np.r_[xy, height]
            if not (-.29 <= xy[0] <= .29 and -.77 <= xy[1] <= -.23):
                continue
            distance = _distance_to_box(target, low, high)
            if not .020 <= distance <= MAX_NEAR_OBJECT_DISTANCE_M:
                continue
            gripper_low, gripper_high = target+lower, target+upper
            if any(np.all(gripper_high >= other_low-.008) and
                   np.all(gripper_low <= other_high+.008)
                   for _, other_low, other_high in bounds):
                continue
            result.append({"anchor_name": name, "target_tcp": target.tolist(),
                           "planned_anchor_aabb_distance_m": distance})
    return result


def _object_distance(env, distance=.020):
    points = [c for body in env.object_ids.values()
              for c in p.getClosestPoints(env.gripper.body, body, distance=distance)]
    return min((float(c[8]) for c in points), default=distance)


def _visibility(env, names):
    from VLA.audit_finetune_data import resized_mask_metrics, mask_contrast
    result = {}
    for view in ("front", "wrist"):
        rgb = env.render(view)
        packed = env._last_segmentation[view]
        ids = np.where(packed >= 0, packed & ((1 << 24)-1), -1)
        result[view] = {}
        for name in names:
            mask = ids == env.object_ids[name]
            pixels, width, height = resized_mask_metrics(mask)
            contrast = mask_contrast(rgb, mask)
            result[view][name] = {"pixels_224": int(pixels), "width_224": int(width),
                                  "height_224": int(height), "contrast": float(contrast),
                                  "passes": bool(pixels >= 48 and min(width, height) >= 7 and contrast >= 25)}
    return result


def validate_closed_empty_near_object_state(env, evidence):
    """Audit/replay gate; no success feedback or target ID is used."""
    from VLA.common.physics_safety import surface_clearance
    if env.safety_latched:
        raise RuntimeError("Near-object start hit an existing physics guard")
    if env.gripper.grasped_object is not None:
        raise RuntimeError("Near-object start must not hold an object")
    if p.getContactPoints(bodyA=env.gripper.body):
        raise RuntimeError("Near-object closed-empty start has physical contacts")
    if any(env.gripper._bilateral(body) for body in env.object_ids.values()):
        raise RuntimeError("Near-object closed-empty start has bilateral object contact")
    angle = float(p.getJointState(env.gripper.body, 1)[0])
    closure = float(env.measured_gripper_closure())
    if angle < MIN_CLOSED_FK*env.gripper.CLOSED_ANGLE or closure < MIN_CLOSED_FK:
        raise RuntimeError("Requested CLOSE is not actual >=95% encoder/FK closure")
    surface = float(surface_clearance(env))
    distance = _object_distance(env, distance=.25)
    if surface < .003 or distance < MIN_OBJECT_CLEARANCE_M:
        raise RuntimeError("Near-object start lacks positive floor/object clearance")
    body = env.object_ids[evidence["anchor_name"]]
    low, high = map(np.asarray, p.getAABB(body))
    tcp = np.asarray(env.gripper.grasp_center())
    near = _distance_to_box(tcp, low, high)
    if not .015 <= near <= MAX_NEAR_OBJECT_DISTANCE_M or tcp[2] > high[2]+.020:
        raise RuntimeError("Empty start is not beside the physical object body")
    return {"actual_closure": closure, "actual_main_angle_rad": angle,
            "grasped_object": None, "gripper_contact_count": 0,
            "bilateral_contact_count": 0, "surface_clearance_m": surface,
            "closest_object_clearance_m": distance, "anchor_aabb_distance_m": near,
            "tcp_world": tcp.tolist(), "anchor_aabb": [low.tolist(), high.tolist()]}


def validated_near_object_reset_environment(env, canonical_task, startup_actions, evidence):
    """Canonical BOTH-text replay with a fresh physical/camera t=0 gate."""
    from VLA.audit_counterfactual_collection import CanonicalSceneResetEnvironment, recovery_visibility_names

    class ValidatedNearObjectReset(CanonicalSceneResetEnvironment):
        def reset(self, task):
            observation = super().reset(task)
            actual = validate_closed_empty_near_object_state(self.env, evidence)
            visible = _visibility(self.env, recovery_visibility_names(canonical_task, self.env.object_ids))
            if not all(cell["passes"] for cell in visible["front"].values()):
                raise RuntimeError("Canonical near-object replay loses front scene visibility")
            if not visible["wrist"][evidence["anchor_name"]]["passes"]:
                raise RuntimeError("Canonical near-object replay loses wrist anchor visibility")
            self.near_object_recovery_replay_evidence = {"actual_state": actual, "visibility": visible}
            return observation

    return ValidatedNearObjectReset(env, canonical_task, startup_actions)


def closed_empty_near_object_start(env, canonical_task, *, audit_callback=None, settling_baseline=None,
                                  bounded_transit=False, expanded_candidates=False):
    """Return canonical startup controls + evidence for a safe actual miss.

    Bounded OPEN vertical raise, high travel, OPEN descent beside an object,
    then CLOSE in free space. Guarded ordinary env.step controls ONLY; no
    teleport/resetJointState, source pose edits, or changes to success logic.
    A failed candidate is reset, not recovered by pushing through geometry.
    """
    from VLA.audit_counterfactual_collection import recovery_visibility_names
    env.reset(canonical_task)
    names = recovery_visibility_names(canonical_task, env.object_ids)
    centre = np.asarray(env.gripper.grasp_center())
    low, high = _gripper_aabb(env)
    bounds = [(name, *map(np.asarray, p.getAABB(body)))
              for name, body in env.object_ids.items()]
    table_top = p.getAABB(env.table_id)[1][2]
    candidates = near_object_candidates(bounds, (low-centre, high-centre), table_top)
    # Miss a plausible requested pick, not only a destination. The FULL list
    # is independent of which object either paired instruction selects.
    allowed_anchors = set(canonical_task.config_pick_items)
    candidates = [candidate for candidate in candidates if candidate["anchor_name"] in allowed_anchors]
    candidates.sort(key=lambda candidate: (candidate["planned_anchor_aabb_distance_m"],
                                          candidate["anchor_name"], candidate["target_tcp"]))
    if expanded_candidates:
        from VLA.near_object_candidate_expansion import supplementary_near_object_candidates
        extra = supplementary_near_object_candidates(bounds, (low-centre, high-centre), table_top)
        extra = [candidate for candidate in extra if candidate["anchor_name"] in allowed_anchors]
        extra.sort(key=lambda candidate: (candidate["planned_anchor_aabb_distance_m"],
                                          candidate["anchor_name"], candidate["target_tcp"]))
        # Preserve the original finite cardinal prefix, then bounded extras.
        candidates = candidates[:32] + extra[:32]
    if not candidates:
        raise RuntimeError("No near-body candidate clears the physical OPEN gripper; do not move source objects")
    previous_render = env.render_observations
    rejected = []
    try:
        env.render_observations = False
        for candidate in candidates[:64 if expanded_candidates else 32]:
            env.reset(canonical_task)
            original = {name: p.getBasePositionAndOrientation(body)
                        for name, body in env.object_ids.items()}
            actions = []
            target = np.asarray(candidate["target_tcp"])
            travel_height = max(.24, max(b[2][2] for b in bounds)+.12,
                                0. if bounded_transit else float(env.gripper.grasp_center()[2]))
            start = np.asarray(env.gripper.grasp_center())
            stages = (np.r_[start[:2], travel_height],
                      np.r_[target[:2], travel_height], target)
            try:
                for stage in stages:
                    for _ in range(35):
                        delta = stage-env.gripper.grasp_center()
                        if np.linalg.norm(delta) < .007:
                            break
                        if len(actions) >= canonical_task.max_steps-16:
                            raise RuntimeError("Bounded startup budget exhausted before close")
                        action = np.r_[np.clip(.65*delta, -.030, .030), np.zeros(3), -1].astype(np.float32)
                        _, _, done, info = env.step(action)
                        actions.append(np.asarray(info.get("executed_canonical_action", action), np.float32))
                        if done or env.gripper.grasped_object is not None or p.getContactPoints(bodyA=env.gripper.body):
                            raise RuntimeError("Startup motion terminated, touched or held; reject rather than push")
                        if _object_distance(env) < MIN_OBJECT_CLEARANCE_M:
                            raise RuntimeError("Startup path approached an object collision")
                    else:
                        raise RuntimeError("Bounded startup stage did not converge")
                close_count = 0
                for _ in range(15):
                    action = np.asarray([0, 0, 0, 0, 0, 0, 1], np.float32)
                    _, _, done, info = env.step(action)
                    actions.append(np.asarray(info.get("executed_canonical_action", action), np.float32))
                    close_count += 1
                    if done or env.gripper.grasped_object is not None or p.getContactPoints(bodyA=env.gripper.body):
                        raise RuntimeError("Missed-grasp CLOSE contacted/held or terminated; reject")
                    if _object_distance(env) < MIN_OBJECT_CLEARANCE_M:
                        raise RuntimeError("Closing fingers approached an object collision")
                    if (p.getJointState(env.gripper.body, 1)[0] >= .95*env.gripper.CLOSED_ANGLE
                            and env.measured_gripper_closure() >= .95):
                        break
                else:
                    raise RuntimeError("15 CLOSE controls did not reach >=95% actual closure")
                evidence = dict(candidate)
                evidence["actual_state"] = validate_closed_empty_near_object_state(env, evidence)
                visibility = _visibility(env, names)
                if audit_callback is not None:
                    # Read-only diagnostic hook BEFORE the original rejection
                    # gates. No control/visibility/drift threshold is changed.
                    audit_callback(env, candidate, actions, original, visibility)
                if not all(cell["passes"] for cell in visibility["front"].values()):
                    raise RuntimeError("Near-object start occludes a configured front-camera scene object")
                if not visibility["wrist"][candidate["anchor_name"]]["passes"]:
                    raise RuntimeError("Near-object miss anchor is not resolved in the wrist model image")
                shift = {name: float(np.linalg.norm(np.asarray(p.getBasePositionAndOrientation(env.object_ids[name])[0])-pose[0]))
                         for name, pose in original.items()}
                rotation_drift = {}
                for name, pose in original.items():
                    before_q = np.asarray(pose[1], dtype=float)
                    after_q = np.asarray(p.getBasePositionAndOrientation(env.object_ids[name])[1], dtype=float)
                    cosine = abs(float(np.dot(before_q, after_q)/(np.linalg.norm(before_q)*np.linalg.norm(after_q))))
                    rotation_drift[name] = float(2*np.arccos(np.clip(cosine, 0, 1)))
                settling = None
                if (max(shift.values(), default=0) > .001 or
                        max(rotation_drift.values(), default=0) > .01):
                    if settling_baseline is None:
                        raise RuntimeError("Source object changed >1mm/0.01rad during contact-free startup")
                    from VLA.settling_baseline import natural_settling_check
                    actual = {name: p.getBasePositionAndOrientation(body) for name, body in env.object_ids.items()}
                    reference = settling_baseline(canonical_task, actions)
                    settling = natural_settling_check(actual, reference)
                    if not settling["passed"]:
                        raise RuntimeError("Source object motion differs from matched-time no-motion baseline")
                evidence.update(visibility=visibility, source_object_translation_drift_m=shift,
                    source_object_rotation_drift_rad=rotation_drift,
                    close_control_count=close_count, startup_control_count=len(actions),
                    startup_controls_are_training_labels=False, source_object_poses_modified=False,
                    safety_guards_modified=False, protocol="near_body_contact_free_attempted_close_v1",
                    scope="Deliberate near-object collision-free missed close; not contact-slip or all natural failed grasps",
                    rejected_candidate_count=len(rejected), rejected_candidates=rejected)
                evidence.update(bounded_object_clear_transit=bool(bounded_transit),
                                expanded_offline_proposals=bool(expanded_candidates),
                                transit_pad_tcp_height_m=float(travel_height))
                if settling is not None:
                    evidence["natural_settling_baseline_check"] = settling
                return actions, evidence
            except RuntimeError as exc:
                rejected.append({**candidate, "reason": str(exc)})
        raise RuntimeError("No safe visible near-object missed-grasp startup: "+str(rejected))
    finally:
        env.render_observations = previous_render
