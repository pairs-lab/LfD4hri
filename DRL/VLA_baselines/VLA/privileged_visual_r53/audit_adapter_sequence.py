"""Official mini-head CPU structural test, NOT actual-checkpoint/GPU proof."""
import argparse
from pathlib import Path

import torch
from VLA.common.artifact_io import atomic_json
from VLA.new_vlas.adapter_loading import import_native


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(53)
    *_,Head,Projector=import_native()
    head=Head(input_dim=16,hidden_dim=16,action_dim=7,num_task_tokens=512,use_pro_version=True).to(torch.bfloat16)
    projector=Projector(llm_dim=16,proprio_dim=75).to(torch.bfloat16)
    identities={k:id(v) for k,v in head.named_parameters()}
    head.num_task_tokens=768
    if identities!={k:id(v) for k,v in head.named_parameters()}:raise ValueError('Parameters changed')
    features=torch.randn(1,25,824,16,dtype=torch.bfloat16,requires_grad=True)
    output=head.predict_action(features,proprio=torch.ones(75),proprio_projector=projector,phase='Inference')
    if output.shape!=(1,8,7) or not torch.isfinite(output).all():raise ValueError('Native horizon/finite differs')
    output.float().square().mean().backward()
    grad=features.grad[:,:,512:768]
    if not torch.isfinite(grad).all() or not torch.count_nonzero(grad):raise ValueError('Third task-view tokens disconnected')
    atomic_json(args.output/'audit.json',{'passed':True,
        'scope':'official_small_dim_CPU_module_shape_gradient_test_NOT_real_checkpoint_training_or_SR',
        'task_tokens':768,'third_task_token_gradient_nonzero':int(torch.count_nonzero(grad)),
        'output_shape':list(output.shape),'learned_parameter_identities_unchanged':True,
        'actual_native_GPU_gradient_and_memory_gate_still_required':True})
    print('Official native Adapter mini-head supports3x256 visual task tokens and third-view gradient.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
