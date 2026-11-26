"""Isolated official VLA-Adapter imports without unused RLDS initializers.

Only package __init__ side effects are bypassed. All model, attention,
projector, token and action-head implementations remain the official files.
Use in a separate interpreter; never override another loaded prismatic tree.
"""
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
from types import ModuleType
from types import MethodType


def install_amp_query_cast(model):
    """Cast only query activations for native indexed embedding assignment.

    Official inference assumes all weights BF16. Keeping trainable FP32 query
    masters requires an activation cast at this indexed-write boundary; the
    differentiable cast preserves gradients and does not alter model math.
    """
    original = model._replace_input_embeddings

    def replace(self, input_embeddings, all_actions_mask, noisy_action_features):
        return original(input_embeddings, all_actions_mask,
                        noisy_action_features.to(input_embeddings.dtype))

    model._replace_input_embeddings = MethodType(replace, model)
    return {'query_master_dtype': 'float32', 'query_activation_cast_to_embedding_dtype': True,
            'native_index_assignment_and_forward_preserved': True}


def import_native(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]/'third_party/vla_adapter'
    packages = ('prismatic', 'prismatic.models', 'prismatic.extern',
                'prismatic.extern.hf', 'prismatic.training', 'prismatic.vla')
    for name in packages:
        path = root/Path(*name.split('.'))
        if name in sys.modules:
            if list(getattr(sys.modules[name], '__path__', [])) != [str(path)]:
                raise ValueError('A different prismatic package is already imported; use an isolated subprocess')
            continue
        package = ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    return (OpenVLAConfig, OpenVLAForActionPrediction, PrismaticImageProcessor,
            PrismaticProcessor, L1RegressionActionHead, ProprioProjector)
