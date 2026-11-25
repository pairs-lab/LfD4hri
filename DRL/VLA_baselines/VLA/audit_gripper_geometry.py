"""Read-only geometric calibration audit of the original Robotiq asset.

Reports actual scaled mesh surfaces rather than mistaking inertial/link COM
distance for inner aperture. No meshes, rendering defaults or sensors change.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
import trimesh

from VLA.common.provenance import sha256_file


def transform_points(points, position, quaternion):
    rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    return np.asarray(points) @ rotation.T + np.asarray(position)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1] / "asset" / "robotiq_2f_85"
    root = root.resolve()
    urdf = root / "robotiq_2f_85.urdf"
    xml = ET.parse(urdf).getroot()
    meshes = {}
    for side in ("right", "left"):
        link = xml.find(f"link[@name='robotiq_2f_85_{side}_pad']")
        for kind in ("visual", "collision"):
            mesh = link.find(f"{kind}/geometry/mesh")
            scale = np.asarray([float(x) for x in mesh.get("scale", "1 1 1").split()])
            path = root / mesh.get("filename")
            loaded = trimesh.load(path, force="mesh", process=False)
            meshes[(side, kind)] = np.asarray(loaded.vertices, dtype=np.float64) * scale
    client = p.connect(p.DIRECT)
    rows = []
    try:
        body = p.loadURDF(str(urdf), useFixedBase=True, physicsClientId=client)
        link_ids = {p.getJointInfo(body, i, physicsClientId=client)[12].decode(): i for i in range(p.getNumJoints(body, physicsClientId=client))}
        for angle in np.linspace(0, .78, 21):
            for joint, value in zip((1, 3, 5, 6, 8, 10), (angle, -angle, angle, angle, -angle, angle), strict=True):
                p.resetJointState(body, joint, float(value), physicsClientId=client)
            states = [p.getLinkState(body, link_ids[f"robotiq_2f_85_{side}_pad"], computeForwardKinematics=True,
                                    physicsClientId=client) for side in ("right", "left")]
            coms = [np.asarray(s[0]) for s in states]
            axis = coms[1] - coms[0]
            axis /= np.linalg.norm(axis)
            row = {"drive_angle_rad": float(angle), "pad_com_distance_m": float(np.linalg.norm(coms[1] - coms[0]))}
            centres = {}
            for kind in ("visual", "collision"):
                worlds = [transform_points(meshes[(side, kind)], s[4], s[5])
                          for side, s in zip(("right", "left"), states, strict=True)]
                projections = [vertices @ axis for vertices in worlds]
                # Right pad -> left pad axis; subtract facing support surfaces.
                row[f"{kind}_inner_support_gap_m"] = float(projections[1].min() - projections[0].max())
                centres[kind] = np.mean([vertices.mean(axis=0) for vertices in worlds], axis=0)
            row["grasp_com_vs_collision_surface_centre_offset_m"] = float(np.linalg.norm(np.mean(coms, axis=0) - centres["collision"]))
            rows.append(row)
    finally:
        p.disconnect(client)
    collision_bounds = {side: {"min_m": meshes[(side, "collision")].min(axis=0).tolist(),
                              "max_m": meshes[(side, "collision")].max(axis=0).tolist()}
                        for side in ("right", "left")}
    visual_extents = np.ptp(meshes[("right", "visual")], axis=0)
    collision_extents = np.ptp(meshes[("right", "collision")], axis=0)
    monotonic = bool(np.all(np.diff([r["collision_inner_support_gap_m"] for r in rows]) < 0))
    report = {"numerical_geometry_passed": monotonic,
              "asset": str(urdf), "asset_sha256": sha256_file(urdf),
              "pad_mesh_sha256": sha256_file(root / "robotiq-2f-pad.stl"),
              "visual_pad_extents_m": visual_extents.tolist(), "collision_pad_extents_m": collision_extents.tolist(),
              "collision_to_visual_scale_ratio": (collision_extents / visual_extents).tolist(),
              "collision_pad_local_bounds": collision_bounds,
              "rows": rows, "collision_gap_decreases_with_closure": monotonic,
              "physical_aperture_calibration_qualified": False,
              "caveats": ["Support projection is a geometric aperture estimate, not measured contact gap or Bullet convex-margin calibration",
                          "Actual follower joint feedback, contacts and pad alignment must be measured during dynamic grasping",
                          "Pad visual scale is 10x smaller than collision; source asset intentionally retained",
                          "Never divide link COM separation by 85mm or silently rewrite historical proprioception"],
              "source_assets_changed": False, "runtime_state_contract_changed": False,
              "scope": "Static asset and FK audit, not learned-policy success or full gripper dynamics qualification"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("rows", "collision_pad_local_bounds")}, indent=2))


if __name__ == "__main__":
    main()
