"""Camera configs."""
import numpy as np
import pybullet as p

# Video-only HD cameras. Keep the legacy RGB-D policy calibration unchanged.
DEMO_VIEWS = (
    {'name': 'front', 'eye': (-.85, -1.85, 1.15),
     'target': (0., -.38, .35), 'up': (0., 0., 1.), 'fov': 47.},
    {'name': 'side', 'eye': (1.25, -1.45, 1.05),
     'target': (0., -.35, .34), 'up': (0., 0., 1.), 'fov': 48.},
    {'name': 'workspace', 'eye': (-.05, -.85, 1.25),
     'target': (0., -.48, .04), 'up': (0., 1., 0.), 'fov': 43.},
)

class Camera():
    """Default configuration with 3 RealSense RGB-D cameras."""
    # Mimic RealSense D415 RGB-D camera parameters.
    image_size = (720, 720)
    side_focal = 450.0
    cx, cy = 320.0, 240.0
    scale = 0.95
    side_intrinsics = (
        side_focal * scale, 0, cx * scale,
        0, side_focal * scale, cy * scale,
        0, 0, 1
    )

    top_focal = 512.0  # proportional to image size for correct FOV
    top_intrinsics = (top_focal, 0, image_size[0] / 2,
                  0, top_focal, image_size[1] / 2,
                  0, 0, 1)
    # Set default camera poses.

    front_position = (0, -1, 0.805)
    front_rotation = (np.pi / 4 - np.pi/45, np.pi, np.pi)
    front_orientation = p.getQuaternionFromEuler(front_rotation)

    diagonal_position = (-0.56, -0.7, 0.82)
    diagonal_rotation = (np.pi / 4 - np.pi/36, np.pi, 2*np.pi/3)
    diagonal_orientation = p.getQuaternionFromEuler(diagonal_rotation)

    side_position = (-0.53, -0.35, 0.80)
    side_rotation = (np.pi / 4 - np.pi/36, np.pi, np.pi/2)
    side_orientation = p.getQuaternionFromEuler(side_rotation)

    # Mirrored counterpart of the user's left-side optical pose.
    right_side_position = (0.53, -0.35, 0.80)
    right_side_rotation = (np.pi / 4 - np.pi/36, np.pi, -np.pi/2)
    right_side_orientation = p.getQuaternionFromEuler(right_side_rotation)

    top_position = (0, -0.5, 0.5)
    top_rotation = (0, np.pi, -np.pi / 2)
    top_orientation = p.getQuaternionFromEuler(top_rotation)

  # Default camera configs.
    CONFIG = [{
      'image_size': image_size,  # front view
      'intrinsics': side_intrinsics,
      'position': front_position,
      'orientation': front_orientation,
      'zrange': (0.01, 10.),
      'noise': False
    }, {
      'image_size': image_size,  # side view
      'intrinsics': side_intrinsics,
      'position': side_position,
      'orientation': side_orientation,
      'zrange': (0.01, 10.),
      'noise': False
    }, {
      'image_size': image_size,  # diagonal view
      'intrinsics': side_intrinsics,
      'position': diagonal_position,
      'orientation': diagonal_orientation,
      'zrange': (0.01, 10.),
      'noise': False
    }, {
      'image_size': image_size,  # top view
      'intrinsics': top_intrinsics,
      'position': top_position,
      'orientation': top_orientation,
      'zrange': (0.01, 10.),
      'noise': False
    }]


def optical_demo_view(name, position, orientation):
    """Use the same optical +Z/-Y convention as Utils.render_image."""
    rotation = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    eye = np.array(position)
    return {'name': name, 'eye': tuple(position),
            'target': tuple(eye + rotation @ [0., 0., 1.]),
            'up': tuple(rotation @ [0., -1., 0.]),
            'fov': float(np.degrees(2*np.arctan(Camera.image_size[0]/(2*Camera.side_intrinsics[0]))))}


# Extra video views do not alter Camera.CONFIG or the policy's RGB-D inputs.
CINEMATIC_VIEWS = DEMO_VIEWS + (
    optical_demo_view('front_user', Camera.front_position, Camera.front_orientation),
    optical_demo_view('left_user', Camera.side_position, Camera.side_orientation),
    optical_demo_view('right_user', Camera.right_side_position, Camera.right_side_orientation),
)
