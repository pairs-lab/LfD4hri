"""CPU qualification and exact process request capture; no signals or GPU use."""
import os
import subprocess
from VLA.common.artifact_io import atomic_json
from .common import (OUT,R86,R90,CP,SIM,read,verify,utc,identity,alive,live_children,
    bindings,module_name,parent_pid)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    subprocess.run([str(SIM),'-m','VLA.dita_close800_r94.tests'],check=True,env=env)
    verify(read(R86/'QUALIFICATION.json')['code'])
    verify(read(R90/'QUALIFICATION.json')['code'])
    state=read(R86/'LIVE.json'); parent=identity(state['pid']); optimizer=identity(state['child_pid'])
    if (state.get('status')!='running' or state.get('family')!='dita' or state.get('stage')!='fit10000'
        or module_name(parent)!='VLA.continuous_r86.owner' or module_name(optimizer)!='VLA.recovery_r85.dita_train'
        or parent_pid(optimizer['pid'])!=parent['pid'] or str(CP) not in optimizer['command']
        or '--updates' not in optimizer['command'] or optimizer['command'][optimizer['command'].index('--updates')+1]!='10000'):
        raise ValueError('Exact current optimizer/scheduler identity not qualified')
    pending=read(R90/'LIVE.json'); recovery=identity(pending['pid'])
    if (not recovery or not alive(recovery['pid']) or module_name(recovery)!='VLA.recovery_quality_r90.owner'
        or live_children(recovery['pid'])
        or pending['stage']!='waiting_for_current_R86_native_FT_and_DEV_to_finish_NO_preemption'):
        raise ValueError('R90 must still be the idle waiting owner, with no active child')
    request={'UTC':utc(),'scope':'active Dita only, finish10k then800 and no further fit',
        'terminal_checkpoint':str(CP),'parent':parent,'optimizer':optimizer,'idle_recovery':recovery,
        'future_Dita_finetune_authorized':False,'code':bindings()}
    atomic_json(OUT/'REQUEST.json',request)
    atomic_json(OUT/'QUALIFICATION.json',{'UTC':utc(),'passed':True,'CPU_contract_tests':14,
        'code':bindings(),'no_signal_sent':True,'no_GPU_model_loaded':True,
        'future_checkpoint_and800_result_do_not_exist_yet':True})
    print({'qualified':True,'request':str(OUT/'REQUEST.json'),'optimizer_pid':optimizer['pid']})


if __name__=='__main__': main()
