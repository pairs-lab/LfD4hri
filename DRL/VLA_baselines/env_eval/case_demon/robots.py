"""Five arms in the DRL workspace frame; assets are read, never modified."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DOWNLOADS = Path('/home/uet/Downloads')
DRL = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RobotSpec:
    label: str
    urdf: Path
    home: tuple
    ee_link: int
    base_yaw: float
    colors: tuple = ()  # (link index or -1, rgba); empty keeps the URDF colours


# The drl_* repos place the workspace on +X. Yawing their fixed base by -90 deg
# gives the DRL -Y workspace, so CINEMATIC_VIEWS stay identical for all arms.
# Panda's .dae files carry no usable colour, so the renderer otherwise paints each
# link from its default palette. Franka shell: off-white, dark wrist flange.
PANDA_COLORS = tuple((link, (.93, .93, .91, 1.)) for link in range(-1, 6))+((6, (.16, .16, .17, 1.)),)
# Kinova Gen3: light anodized silver with a dark bracelet.
KINOVA_COLORS = tuple((link, (.82, .84, .86, 1.)) for link in range(-1, 6))+((6, (.14, .14, .15, 1.)),)
ROBOTS = {
    'Panda': RobotSpec('Panda', DOWNLOADS / 'drl_panda/asset/panda/panda.urdf',
                       (0, -np.pi/6, 0, -2*np.pi/3, 0, np.pi/2, 0), 6, -np.pi/2, PANDA_COLORS),
    'KUKA': RobotSpec('KUKA', DOWNLOADS / 'drl_kuka/asset/kuka/kuka.urdf',
                      (0, -np.pi/8, 0, -3*np.pi/5, 0, 8*np.pi/15, 0), 6, -np.pi/2),
    'Kinova': RobotSpec('Kinova', DOWNLOADS / 'drl_kinova/asset/kinova/kinova.urdf',
                        (0, 0, 0, np.pi/2, 0, np.pi/2, 0), 7, -np.pi/2, KINOVA_COLORS),
    'UR5e': RobotSpec('UR5e', DRL / 'asset/ur5e/ur5e.urdf',
                      (np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, 3*np.pi/2, 0), 9, 0.),
    'UF850': RobotSpec('UF850', DRL / 'asset/uf850/uf850.urdf',
                       (-np.pi/2, -np.pi/15, -2*np.pi/9, np.pi, 2*np.pi/7, 0), 6, 0.),
}
