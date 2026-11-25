"""Offline finite near-body proposal coverage; no selected object or text.

The original cardinal rule is preserved by its caller. These supplementary
diagonal/tangential proposals retain the same whole OPEN-gripper footprint,
table margin, workspace and near-body distance/height constraints.
"""
import numpy as np

from VLA.near_object_recovery_start import MAX_NEAR_OBJECT_DISTANCE_M, _distance_to_box


def supplementary_near_object_candidates(bounds, offsets, table_top):
    lower, upper = map(np.asarray, offsets)
    result = []
    for name, low, high in sorted(bounds, key=lambda item: item[0]):
        low, high = np.asarray(low), np.asarray(high)
        centre = (low + high) / 2
        height = max(float(centre[2]), float(table_top - lower[2] + .010))
        if height > high[2] + .020:
            continue
        left, right = low[0] - upper[0] - .010, high[0] - lower[0] + .010
        front, back = low[1] - upper[1] - .010, high[1] - lower[1] + .010
        proposals = [(x, y) for x in (left, right) for y in (front, back)]
        for offset in (-.040, .040):
            proposals.extend([(left, centre[1] + offset), (right, centre[1] + offset),
                              (centre[0] + offset, front), (centre[0] + offset, back)])
        for xy in proposals:
            target = np.r_[xy, height]
            if not (-.29 <= xy[0] <= .29 and -.77 <= xy[1] <= -.23):
                continue
            distance = _distance_to_box(target, low, high)
            if not .020 <= distance <= MAX_NEAR_OBJECT_DISTANCE_M:
                continue
            gripper_low, gripper_high = target + lower, target + upper
            if any(np.all(gripper_high >= other_low - .008) and
                   np.all(gripper_low <= other_high + .008)
                   for _, other_low, other_high in bounds):
                continue
            result.append({'anchor_name': name, 'target_tcp': target.tolist(),
                           'planned_anchor_aabb_distance_m': distance,
                           'proposal_family': 'bounded_diagonal_tangential_all_objects_v1'})
    return result
