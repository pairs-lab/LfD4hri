"""Same neural loop/plant; version-matched receipt and real safety diagnostics."""
import argparse
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from .audit import read


def renderer(family):
    if family == 'dita':
        # The native Dita process uses NumPy1.26.4, evaluator uses2.2.6.
        # Correct ONLY its receipt/reference pixels, never the policy image.
        from VLA.receipt_recovery_r88.map_reference import render
    elif family == 'vlanext_s':
        # S and its evaluator both use NumPy2.2.6; do not apply Dita's
        # compatibility arithmetic blindly to this distinct native process.
        from VLA.privileged_visual_r53.scene_map import render
    else:
        raise ValueError('Unqualified model family')
    return render


def diagnostic_class(base, output):
    class Diagnosed(base):
        def __init__(self, *args, **kwargs):
            self._diagnostic_seed = kwargs.get('seed')
            super().__init__(*args, **kwargs)

        def step(self, action):
            result = super().step(action)
            info = result[3]
            if info.get('safety_violation'):
                atomic_json(Path(output)/'SAFETY_DIAGNOSTICS'/f'seed{self._diagnostic_seed}'/f'{self.task.name}_{self.step_count}.json', {
                    'task': self.task.name, 'prompt': self.task.prompt, 'step': self.step_count,
                    'requested_action': info.get('requested_action'),
                    'executed_canonical_action': info.get('executed_canonical_action'),
                    'safety_violation': info['safety_violation'],
                    'before_rollback': info.get('safety_diagnostics_before_rollback'),
                    'rollback_restored': info.get('rollback_restored'),
                    'audit_only_not_policy_observation_or_action_override': True})
            return result
    return Diagnosed


def main(a):
    from VLA.autonomy_r83 import evaluate as old
    from VLA.continuous_r86.evaluate import main as actual_main
    from VLA.privileged_visual_r53 import scene_map
    method = read(a.method)
    original_reference, original_map = old.render, scene_map.render
    old.render = renderer(method['family'])
    scene_map.render = old.render
    factory = old.environment_class_for_profile
    old.environment_class_for_profile = lambda profile: diagnostic_class(factory(profile), a.output)
    try:
        actual_main(a)
    finally:
        old.environment_class_for_profile = factory
        old.render, scene_map.render = original_reference, original_map


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('method','cohort','output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    main(p.parse_args())
