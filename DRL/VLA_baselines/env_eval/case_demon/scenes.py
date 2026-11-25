"""Fifteen distinct PUT scenes: one set of three for each robot, none shared."""
from dataclasses import dataclass

import numpy as np

from env_eval.case_demon.robots import ROBOTS

X, Y = (-.30, .30), (-.74, -.29)
# (target, destination bowl) per environment, in ROBOTS order.
TASKS = [
    ('red block', 'blue bowl'), ('red apple', 'green bowl'), ('cyan block', 'orange bowl'),
    ('orange orange', 'yellow bowl'), ('purple block', 'pink bowl'), ('yellow lemon', 'cyan bowl'),
    ('green block', 'red bowl'), ('pink peach', 'brown bowl'), ('brown block', 'gray bowl'),
    ('green pear', 'purple bowl'), ('yellow block', 'blue bowl'), ('red strawberry', 'orange bowl'),
    ('gray block', 'green bowl'), ('green tennis_ball', 'red bowl'), ('orange block', 'yellow bowl'),
]
BLOCKS = ['blue', 'red', 'green', 'orange', 'yellow', 'purple', 'pink', 'cyan', 'brown', 'gray']
FRUITS = ['red apple', 'orange orange', 'yellow lemon', 'pink peach', 'green pear', 'red strawberry']
BOWLS = [c + ' bowl' for c in BLOCKS]


@dataclass(frozen=True)
class Scene:
    scene_id: int
    robot: str
    env: int
    candidate: int
    target: str
    goal: str
    objects: tuple  # (name, (x, y)); first two are target and destination


def radius(name):
    return .095 if name.endswith(' bowl') else .04


def _pack(rng, target, goal, distractors):
    pick = rng.uniform((X[0]+.04, Y[0]+.04), (X[1]-.04, Y[1]-.04))
    place = rng.uniform((X[0]+.095, Y[0]+.095), (X[1]-.095, Y[1]-.095))
    if not .22 < np.linalg.norm(pick-place) < .42:
        return None
    placed = [(target, pick), (goal, place)]
    for name in distractors:
        r = radius(name)
        for _ in range(300):
            xy = rng.uniform((X[0]+r, Y[0]+r), (X[1]-r, Y[1]-r))
            if all(np.linalg.norm(xy-pos) > r+radius(other)+.015 for other, pos in placed):
                placed.append((name, xy))
                break
        else:
            return None
    return placed


def build(scene_id, candidate=0):
    """Deterministic layout; a new candidate only re-packs the same task."""
    if not 1 <= scene_id <= len(TASKS):
        raise ValueError(f'Unknown scene {scene_id}')
    robots = list(ROBOTS)
    robot, env = robots[(scene_id-1)//3], (scene_id-1) % 3 + 1
    target, goal = TASKS[scene_id-1]
    rng = np.random.default_rng(2026110600 + 100*scene_id + candidate)
    others = [c + ' block' for c in BLOCKS if c + ' block' != target]
    fruits = [f for f in FRUITS if f != target]
    rng.shuffle(others), rng.shuffle(fruits)
    bowl2 = str(rng.choice([b for b in BOWLS if b != goal]))
    distractors = [bowl2] + fruits[:1 + scene_id % 2] + others[:3 + scene_id % 3]
    for _ in range(2000):
        placed = _pack(rng, target, goal, distractors)
        if placed:
            break
    else:
        raise ValueError(f'Packing failed: scene {scene_id}, candidate {candidate}')
    objects = tuple((n, tuple(round(float(v), 4) for v in xy)) for n, xy in placed)
    return Scene(scene_id, robot, env, candidate, target, goal, objects)
