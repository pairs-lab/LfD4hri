"""Original neural server; bounded allocator and a fresh prediction parity gate."""
import argparse
from types import SimpleNamespace
import numpy as np
import torch
from VLA.common.artifact_io import atomic_json
from VLA.stable_native_r44.conditioning import install
from .common import CP, read


def main(args):
    # An allocator ceiling only: no casting/quantization or changed layers.
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=0)
    import VLA.privileged_visual_r53.server as original
    create = original.create_runtime
    cfg = read(CP/'config.json')
    def audited_create(family, stats):
        runtime = create(family, stats)
        to = runtime.to
        def audited_to(device):
            result = to(device)
            if str(device) == 'cuda':
                install()
                from VLA.privileged_visual_r53.data import MapData
                data = MapData(cfg['data_root'])
                probe = cfg['probe']
                actual = runtime.predict(data.sample(probe['episode'],probe['frame'],family))
                expected = np.load(CP/'reload_reference.npy', allow_pickle=False)
                error = float(np.max(np.abs(actual-expected)))
                if actual.shape != expected.shape or not np.isfinite(actual).all() or error > 1e-6:
                    raise ValueError('Exact checkpoint fresh native prediction parity failed')
                atomic_json(args.proof, {'passed': True, 'checkpoint':str(CP),
                    'native_prediction_max_abs_error':error, 'native_visual_receipt':runtime.r53_native_visual_receipt,
                    'memory_fraction_ceiling_only_no_model_dtype_or_architecture_change':args.memory_fraction,
                    'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
                    'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20})
            return result
        runtime.to = audited_to
        return runtime
    original.create_runtime = audited_create
    original.main(SimpleNamespace(checkpoint=CP, port=args.port))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True)
    p.add_argument('--memory-fraction',type=float,required=True);p.add_argument('--proof',type=__import__('pathlib').Path,required=True)
    main(p.parse_args())
