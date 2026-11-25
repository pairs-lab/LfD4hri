"""Deterministic task definitions derived from the existing RL primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping

import numpy as np

from config import BOUNDS, INSTRUCTION_FORM, PICK_TARGETS, PLACE_TARGETS

from .spatial import FIXED_DESTINATION


@dataclass(frozen=True)
class SceneObjectSpec:
    """A non-target object rendered in the same scene as the commanded task."""

    name: str
    position: tuple[float, float, float]
    fixed: bool
    source_role: str


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    name: str
    prompt: str
    pick_name: str
    pick_position: tuple[float, float, float]
    place_name: str | None = None
    place_position: tuple[float, float, float] | None = None
    max_steps: int = 120
    scene_profile: str = "single_target"
    scene_objects: tuple[SceneObjectSpec, ...] = ()
    config_pick_items: tuple[str, ...] = ()
    config_place_items: tuple[str, ...] = ()
    config_selected_place: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "TaskSpec":
        data = dict(values)
        data["pick_position"] = tuple(data["pick_position"])
        if data.get("place_position") is not None:
            data["place_position"] = tuple(data["place_position"])
        data["scene_objects"] = tuple(
            SceneObjectSpec(
                name=item["name"],
                position=tuple(item["position"]),
                fixed=bool(item["fixed"]),
                source_role=item["source_role"],
            )
            for item in data.get("scene_objects", ())
        )
        data["config_pick_items"] = tuple(data.get("config_pick_items", ()))
        data["config_place_items"] = tuple(data.get("config_place_items", ()))
        return cls(**data)


# Test 1 intentionally uses the same object vocabulary and workspace as config.py,
# while fixing initial poses so every policy receives byte-for-byte equivalent
# trials.  Blocks make physical Robotiq grasps reproducible; the place task also
# uses the existing bowl asset.
TASKS: Mapping[str, TaskSpec] = {
    "touch": TaskSpec(
        task_id=0,
        name="touch",
        prompt="Touch the red block",
        pick_name="red block",
        pick_position=(-0.12, -0.52, 0.021),
        max_steps=80,
    ),
    "pick": TaskSpec(
        task_id=1,
        name="pick",
        prompt="Pick the blue block up",
        pick_name="blue block",
        pick_position=(-0.10, -0.53, 0.021),
        max_steps=120,
    ),
    "move": TaskSpec(
        task_id=2,
        name="move",
        prompt="Move the green block to the top right corner",
        pick_name="green block",
        pick_position=(-0.13, -0.58, 0.021),
        place_name="top right corner",
        place_position=(0.25, -0.25, 0.0),
        max_steps=160,
    ),
    "pick_place": TaskSpec(
        task_id=3,
        name="pick_place",
        prompt="Put the yellow block onto the blue bowl",
        pick_name="yellow block",
        pick_position=(-0.13, -0.58, 0.021),
        place_name="blue bowl",
        place_position=(0.14, -0.40, 0.0),
        max_steps=200,
    ),
}


def select_tasks(selection: str) -> list[TaskSpec]:
    if selection == "all":
        return list(TASKS.values())
    names = [item.strip() for item in selection.split(",") if item.strip()]
    unknown = [name for name in names if name not in TASKS]
    if unknown:
        raise ValueError(f"Unknown task(s) {unknown}; choose from {list(TASKS)} or 'all'")
    return [TASKS[name] for name in names]


def perturb_task(
    task: TaskSpec,
    rng: np.random.Generator,
    magnitude: float,
) -> TaskSpec:
    """Apply the same bounded XY randomization for collection and evaluation.

    The move target is a named fixed workspace corner.  The bowl in pick/place
    is a physical scene object, so it is randomized independently from the
    picked block.  No perturbed coordinates are exposed to a learned policy.
    """
    if magnitude <= 0:
        return task
    pick = np.asarray(task.pick_position, dtype=np.float64)
    pick[:2] += rng.uniform(-magnitude, magnitude, 2)
    place = task.place_position
    if place is not None and task.place_name == "blue bowl":
        place_array = np.asarray(place, dtype=np.float64)
        place_array[:2] += rng.uniform(-magnitude, magnitude, 2)
        place = tuple(float(value) for value in place_array)
    return replace(
        task,
        pick_position=tuple(float(value) for value in pick),
        place_position=place,
    )


def randomize_rl_scene(
    task: TaskSpec,
    rng: np.random.Generator,
    forced_pick_name: str | None = None,
    forced_place_name: str | None = None,
) -> TaskSpec:
    """Reproduce the object-count, asset pool, workspace and spacing of ``random_env``.

    Both the commanded object and (when used) destination are selected from
    the sampled config exactly like ``RobotEnv.reset``.  Coordinates stay in
    simulator metadata and are never exposed through :class:`VLAObservation`.
    """

    pick_count = int(rng.integers(3, 5))
    place_count = int(rng.integers(2, 4))

    if forced_pick_name is not None and forced_pick_name not in PICK_TARGETS:
        raise ValueError(f"Unknown forced pick target {forced_pick_name!r}")
    # random_env removes every configured pick item from the place pool.  When
    # stratifying a destination, exclude it from the entire pick config (not
    # merely from the commanded pick target) so that invariant cannot be
    # violated by a distractor sample.
    reserved = {name for name in (forced_pick_name, forced_place_name) if name is not None}
    pick_candidates = [name for name in PICK_TARGETS if name not in reserved]
    pick_items = rng.choice(
        pick_candidates,
        size=pick_count - int(forced_pick_name is not None),
        replace=False,
    ).tolist()
    if forced_pick_name is not None:
        pick_items.append(forced_pick_name)
        rng.shuffle(pick_items)
    place_candidates = [
        name
        for name in PLACE_TARGETS
        if name not in pick_items
    ]
    if forced_place_name is not None and forced_place_name not in place_candidates:
        raise ValueError(
            f"Forced place target {forced_place_name!r} is unavailable or overlaps pick config"
        )
    remaining_places = [name for name in place_candidates if name != forced_place_name]
    place_items = rng.choice(
        remaining_places,
        size=place_count - int(forced_place_name is not None),
        replace=False,
    ).tolist()
    if forced_place_name is not None:
        place_items.append(forced_place_name)
        rng.shuffle(place_items)
    pick_name = forced_pick_name or str(rng.choice(pick_items))
    selected_place = forced_place_name or str(rng.choice(place_items))
    required_place = selected_place if task.name in {"move", "pick_place"} else None

    physical_names = [
        name
        for name in (*pick_items, *place_items)
        if name not in FIXED_DESTINATION
    ]
    reserved_positions = ()
    if required_place in FIXED_DESTINATION:
        # random_env enforces 13 cm between physical objects but can put a
        # large bowl directly over the subsequently selected named goal.  Such
        # a scene has no collision-free solution for any policy.  Condition
        # the shared benchmark (all RL/VLA policies) on the commanded fixed
        # destination being free by applying the same 13 cm clearance rule.
        reserved_positions = (FIXED_DESTINATION[required_place],)
    positions = _sample_spaced_positions(
        rng,
        len(physical_names),
        minimum_spacing=0.13,
        reserved_positions=reserved_positions,
    )
    position_by_name = {
        name: (
            position[0],
            position[1],
            0.03 if name.endswith((" block", " egg")) else 0.0,
        )
        for name, position in zip(physical_names, positions, strict=True)
    }

    pick_position = position_by_name[pick_name]
    place_position = None
    if required_place is not None:
        place_position = (
            tuple(float(value) for value in FIXED_DESTINATION[required_place])
            if required_place in FIXED_DESTINATION
            else position_by_name[required_place]
        )

    distractors: list[SceneObjectSpec] = []
    for source_role, names in (("pick", pick_items), ("place", place_items)):
        for name in names:
            if name in FIXED_DESTINATION or name in {pick_name, required_place}:
                continue
            distractors.append(
                SceneObjectSpec(
                    name=name,
                    position=position_by_name[name],
                    fixed=name.endswith(" bowl"),
                    source_role=source_role,
                )
            )

    instruction = INSTRUCTION_FORM[task.task_id]
    prompt = (
        instruction.format(pick_name, selected_place)
        if instruction.count("{}") == 2
        else instruction.format(pick_name)
    )
    return replace(
        task,
        prompt=prompt,
        pick_name=pick_name,
        pick_position=pick_position,
        place_name=required_place,
        place_position=place_position,
        scene_profile="rl_random_env",
        scene_objects=tuple(distractors),
        config_pick_items=tuple(pick_items),
        config_place_items=tuple(place_items),
        config_selected_place=selected_place,
    )


def _sample_spaced_positions(
    rng: np.random.Generator,
    count: int,
    minimum_spacing: float,
    reserved_positions: tuple[tuple[float, float, float], ...] = (),
) -> list[tuple[float, float, float]]:
    """Match ``RobotEnv._load``: uniform BOUNDS with at least 13 cm XY spacing."""

    reserved = [np.asarray(position, dtype=np.float64) for position in reserved_positions]
    result: list[np.ndarray] = []
    for _ in range(count):
        for _attempt in range(20_000):
            candidate = np.array(
                (
                    rng.uniform(float(BOUNDS[0, 0]), float(BOUNDS[0, 1])),
                    rng.uniform(float(BOUNDS[1, 0]), float(BOUNDS[1, 1])),
                    0.03,
                ),
                dtype=np.float64,
            )
            prior_positions = [*reserved, *result]
            if not prior_positions or min(
                np.linalg.norm(candidate[:2] - prior[:2]) for prior in prior_positions
            ) > minimum_spacing:
                result.append(candidate)
                break
        else:
            raise RuntimeError(
                f"Could not place {count} scene objects {minimum_spacing} m apart inside BOUNDS"
            )
    return [tuple(float(value) for value in point) for point in result]
