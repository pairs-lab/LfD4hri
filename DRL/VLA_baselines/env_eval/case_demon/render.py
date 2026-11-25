"""Video output in the views346 look: 1080p cinematic view + top-corner close-up."""
import numpy as np
import pybullet as p
import cv2
import imageio_ffmpeg

from utils.camera import CINEMATIC_VIEWS

WIDTH, HEIGHT, FPS = 1920, 1080, 30
TICKS_PER_FRAME = 8  # 240 Hz physics -> real-time 30 FPS playback
VIEWS = {3: CINEMATIC_VIEWS[2], 4: CINEMATIC_VIEWS[3], 6: CINEMATIC_VIEWS[5]}
MACRO_SIZE = (1280, 720)
FRACTIONS = (.31, .28, .25)
MARGIN = round(WIDTH*.018)
TOP = MARGIN+2


def frame(env, cfg, width, height, segmentation=False):
    view = p.computeViewMatrix(cfg['eye'], cfg['target'], cfg.get('up', (0, 0, 1)))
    projection = p.computeProjectionMatrixFOV(cfg['fov'], width/height, .015, 4.)
    flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK
    _, _, rgba, _, seg = p.getCameraImage(width, height, viewMatrix=view, projectionMatrix=projection,
        renderer=env.renderer, shadow=1, lightDirection=[-1, -.6, 2], lightColor=[1, 1, .98],
        lightAmbientCoeff=.55, lightDiffuseCoeff=.65, lightSpecularCoeff=.15, flags=flags)
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3].copy()
    if segmentation:
        return rgb, np.asarray(seg, dtype=np.int64).reshape(height, width)
    return rgb, None


def rectangle(fraction, side):
    width = int(WIDTH*fraction)//2*2
    height = round(width*9/16)//2*2
    left = MARGIN if side == 'left' else WIDTH-width-MARGIN
    return left, TOP, width, height


class Macro:
    """World-stabilized director camera on the tool tip and the target."""
    def __init__(self, env):
        self.env = env
        self.focus = None
        self.framing = None
        self.normal, self.axis = np.array([0., -1., 0.]), np.array([-1., 0., 0.])

    def config(self, phase):
        e = self.env
        tip = e.tip()
        low, high = map(np.asarray, p.getAABB(e.pick))
        obj = (low+high)/2
        separation = float(np.linalg.norm(obj-tip))
        placing = phase in ('release', 'retract', 'settle')
        # As the arm leaves, settle on the placed object instead of chasing the tip.
        blend = max(.65, float(np.clip((separation-.08)/.20, 0, 1))) if placing else .4*float(np.clip((.17-separation)/.10, 0, 1))
        focus = (1-blend)*tip+blend*obj+[0, 0, .025 if placing else .03]
        self.focus = focus.copy() if self.focus is None else self.focus+.28*(focus-self.focus)
        reach = min(separation, .12) if placing else separation
        distance = .32+max(0., reach-.07)*(.8 if blend else .25)
        eye = self.focus+self.normal*distance+self.axis*.055+[0, 0, .105]
        corners = lambda a, b: np.array([[x, y, z] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])])
        head_low, head_high = map(np.asarray, p.getAABB(e.tool_bodies[0]))
        context = corners(low, high) if placing and separation > .2 else np.vstack([corners(low, high), corners(head_low, head_high)])
        ray = eye-self.focus
        length = np.linalg.norm(ray)
        direction = ray/length
        safe = length
        for _ in range(10):
            forward = -direction
            right = np.cross(forward, [0, 0, 1])
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            relative = context-(self.focus+direction*safe)
            depth = relative@forward
            if depth.min() <= .015:
                safe *= 1.3
                continue
            tan = np.tan(np.radians(43.)/2)
            required = max(float(np.max(abs(relative@right)/(depth*tan*(16/9))/.84)),
                           float(np.max(abs(relative@up)/(depth*tan)/.84)))
            if required <= 1:
                break
            safe *= max(1.02, required)
        weight = 1. if phase in ('attach', 'carry', 'lower', 'release', 'retract', 'settle') else float(np.clip((.18-separation)/.10, 0, 1))
        desired = length+weight*(safe-length)
        self.framing = desired if self.framing is None else max(desired, self.framing+.12*(desired-self.framing))
        return {'eye': (self.focus+direction*self.framing).tolist(), 'target': self.focus.tolist(),
                'up': (0, 0, 1), 'fov': 43.}


class InsetPlanner:
    """Measure, over the whole episode, what each candidate corner would cover."""
    SCALE = 4

    def __init__(self, env):
        self.env = env
        self.costs = {v: {(f, s): [0, 0] for f in FRACTIONS for s in ('left', 'right')} for v in VIEWS}

    def __call__(self, env, *_):
        if env.ticks % (TICKS_PER_FRAME*2):
            return
        w, h = WIDTH//self.SCALE, HEIGHT//self.SCALE
        for view, cfg in VIEWS.items():
            _, seg = frame(env, cfg, w, h, segmentation=True)
            ids = seg & ((1 << 24)-1)
            key = (ids == env.pick) | (ids == env.place)
            robot = ids == env.robot
            for (fraction, side), cost in self.costs[view].items():
                left, top, width, height = rectangle(fraction, side)
                box = (slice((top-3)//self.SCALE, (top+height+3)//self.SCALE+1),
                       slice((left-3)//self.SCALE, (left+width+3)//self.SCALE+1))
                cost[0] += int(key[box].sum())
                cost[1] += int(robot[box].sum())

    def choose(self):
        out = {}
        for view, costs in self.costs.items():
            clear = [(f, s) for (f, s), c in costs.items() if c[0] == 0]
            if clear:
                largest = max(f for f, _ in clear)
                fraction, side = min(((f, s) for f, s in clear if f == largest), key=lambda k: costs[k][1])
            else:
                fraction, side = min(costs, key=lambda k: (costs[k][0], k[0]))
            out[str(view)] = {'fraction': fraction, 'side': side, 'key_object_pixels': costs[(fraction, side)][0],
                              'arm_pixels': costs[(fraction, side)][1]}
        return out


class VideoRecorder:
    def __init__(self, env, rollout, directory, insets):
        self.env, self.rollout = env, rollout
        self.macro = Macro(env)
        self.insets = {int(v): rectangle(c['fraction'], c['side']) for v, c in insets.items()}
        self.paths = {v: directory/f'view_{v}.partial.mp4' for v in VIEWS}
        self.final = {v: directory/f'view_{v}.mp4' for v in VIEWS}
        self.writers = {}
        self.frames = 0
        self.samples = {v: [] for v in VIEWS}
        directory.mkdir(parents=True, exist_ok=True)
        for view, path in self.paths.items():
            writer = imageio_ffmpeg.write_frames(str(path), (WIDTH, HEIGHT), fps=FPS, codec='libx264', quality=8,
                macro_block_size=1, ffmpeg_log_level='error',
                output_params=['-preset', 'fast', '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
            writer.send(None)
            self.writers[view] = writer

    def capture(self):
        cfg = self.macro.config(self.rollout.phase)
        macro, _ = frame(self.env, cfg, *MACRO_SIZE)
        for view, camera in VIEWS.items():
            wide, _ = frame(self.env, camera, WIDTH, HEIGHT)
            left, top, width, height = self.insets[view]
            inset = cv2.resize(macro, (width, height), interpolation=cv2.INTER_AREA)
            cv2.rectangle(wide, (left-3, top-3), (left+width+3, top+height+3), (28, 32, 38), -1)
            cv2.rectangle(wide, (left-2, top-2), (left+width+2, top+height+2), (226, 231, 236), 2)
            wide[top:top+height, left:left+width] = inset
            self.writers[view].send(wide.tobytes())
        self.frames += 1

    def __call__(self, env, *_):
        if env.ticks % TICKS_PER_FRAME == 0:
            self.capture()

    def close(self, keep):
        for view, writer in self.writers.items():
            writer.close()
            if keep:
                self.paths[view].replace(self.final[view])
            else:
                self.paths[view].unlink(missing_ok=True)
