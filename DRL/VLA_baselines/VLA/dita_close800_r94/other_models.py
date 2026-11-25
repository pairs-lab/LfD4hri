"""Preserve the previously authorized S-only recovery recipe; Dita is excluded."""
import fcntl
from pathlib import Path
import traceback

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import ROOT, OUT, R86, R90, read, verify, utc, bindings

OTHER=OUT/'other_models'
RECOVERY_FAMILIES=('vlanext_s',)


def main():
    if 'dita' in RECOVERY_FAMILIES: raise ValueError('Closed Dita must never reenter the recovery queue')
    if not (OUT/'CERTIFIED_RESULT.json').exists(): raise ValueError('Terminal Dita benchmark must finish first')
    OTHER.mkdir(exist_ok=True)
    with (OTHER/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OTHER/'STARTED.json').exists(): raise FileExistsError('No duplicate other-model continuation')
        verify(read(OUT/'QUALIFICATION.json')['code'])
        import VLA.recovery_quality_r90.owner as recovery
        import VLA.continuous_r86.owner as old
        original_recovery_code=recovery.bindings()
        if original_recovery_code!=read(R90/'QUALIFICATION.json')['code']:
            raise ValueError('Original recovery source changed; do not bypass data qualification')
        old.OUT=OTHER; old.prior.flow.OUT=OTHER
        old.prior.flow.PORT=5996
        (OTHER/'vlanext_s').mkdir(exist_ok=True)
        atomic_json(OTHER/'STARTED.json',{'UTC':utc(),'families':list(RECOVERY_FAMILIES),
            'Dita_permanently_excluded_from_this_queue':True,'code':bindings()})

        class Flow(old.Flow):
            def __init__(self):
                super().__init__(); self.bound.update(original_recovery_code); self.bound.update(bindings())
            def run(self,python,module,args,label,**kwargs):
                if module in ('VLA.autonomy_r83.evaluate','VLA.recovery_r85.evaluate','VLA.continuous_r86.evaluate'):
                    module='VLA.recovery_quality_r90.evaluate'
                return super().run(python,module,args,label,**kwargs)
            def method(self,checkpoint,name,**kwargs):
                path=super().method(checkpoint,name,**kwargs); value=read(path)
                value['code'].update(original_recovery_code)
                value['code'].update(bindings())
                value['code'].update(read(checkpoint/'TRAINING.json')['code'])
                value['R90_native_loss_and_action_timing_unchanged']=True
                value['R90_data_receipt']=str(R90/'qualified_recovery/ADMISSION.json')
                value['Dita_closed_no_further_Dita_training']=True
                atomic_json(path,value); return path

        owner=Flow(); parent=R86/'vlanext_s/fit8192'; python=old.prior.flow.PY
        try:
            selection=read(R86/'vlanext_s/DEV_SELECTION.json')
            if any(sum(v['successes'] for v in option['per_action'].values())>=24
                and min(v['successes'] for v in option['per_action'].values())>=4
                for option in selection['actual_options']):
                owner.stage='existing_DEV_already_qualified_no_unnecessary_fit'; owner.status('not_retrained'); return
            old.qualified(parent)
            # Same R90 data gate and finite S recipe, no Dita branch or trainer.
            admission=read(R90/'qualified_recovery/ADMISSION.json')
            if not admission['passed'] or admission['split']!='train':
                raise ValueError('Original qualified recovery TRAIN data required')
            verify(admission['binding'])
            owner.resource()
            gate=OTHER/'vlanext_s/recovery_gate16'
            owner.run(python,'VLA.recovery_quality_r90.train',['--family','vlanext_s','fit','--output',gate,
                '--updates',8208,'--accumulation',4,'--resume',parent],'recovery_gate16',hours=1)
            owner.run(python,'VLA.recovery_quality_r90.train',['--family','vlanext_s','reload','--checkpoint',gate],
                'recovery_gate_reload',hours=1)
            cp=OTHER/'vlanext_s/recovery_fit9216'
            owner.run(python,'VLA.recovery_quality_r90.train',['--family','vlanext_s','fit','--output',cp,
                '--updates',9216,'--accumulation',16,'--resume',gate],'recovery_fit9216',hours=24)
            owner.run(python,'VLA.recovery_quality_r90.train',['--family','vlanext_s','reload','--checkpoint',cp],
                'recovery_fit9216_reload',hours=1)
            method=owner.method(cp,'recovery_method')
            dev=owner.evaluate(method,'recovery_DEV40','development_selection',list(range(20283001,20283011)))
            from VLA.recovery_quality_r90.audit import development_audit
            atomic_json(OTHER/'vlanext_s/RECOVERY_DEV_DIAGNOSTIC.json',
                development_audit(OTHER/'vlanext_s/recovery_DEV40/progress.json'))
            if not old.ready(dev):
                owner.stage='recovery_DEV_not_promoted_NO_blind800'; owner.status('needs_further_audit'); return
            conf=owner.evaluate(method,'recovery_CONF40','development_confirmation',list(range(20293101,20293111)))
            if not old.ready(conf):
                owner.stage='recovery_confirmation_not_passed'; owner.status('not_promoted'); return
            method=owner.method(cp,'FROZEN_RECOVERY_METHOD',frozen=True)
            atomic_json(OTHER/'vlanext_s/FROZEN_FINAL.json',{'checkpoint':str(cp),'method':str(method),
                'method_sha256':sha256_file(method),'confirmation_and_final_not_used_to_tune':True})
            final=owner.evaluate(method,'recovery_final800','fresh_secondary_final800',list(range(20294001,20294201)))
            if final['completed_trials']!=800 or any(v['n']!=200 for v in final['per_action'].values()):
                raise ValueError('Actual800 coverage mismatch')
            atomic_json(OTHER/'vlanext_s/FINAL_AUDIT.json',{'passed':True,'actual_trials':800,
                'per_action':final['per_action'],'actual_cohorts_of800':1,'not_five_measured_seeds':True})
            owner.stage='other_model_frozen800_completed'; owner.status('completed')
        except BaseException as error:
            atomic_json(OTHER/'vlanext_s/INTEGRATION_ERROR.json',{'UTC':utc(),'error':str(error),
                'traceback':traceback.format_exc(),'NOT_model_SR':True})
            owner.status('quarantined_other_model_evidence_retained',error=str(error)); raise
        finally:
            owner.stop()


if __name__=='__main__': main()
