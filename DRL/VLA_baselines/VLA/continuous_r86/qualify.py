"""Qualification ledger and exact prior-owner identity before daemon launch."""
import unittest
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.recovery_r85.common import OUT as R85,read,utc,verify
from VLA.replacement_r84.common import bindings as native_bindings
from VLA.dita_transfer_r84.common import bindings as dita_bindings
from .tests import Contracts
from .owner import OUT,bindings,identity,alive


if __name__=='__main__':
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful():raise ValueError('Do not start a daemon after failing repair/safety tests')
    evidence={}
    for path in (OUT/'FULL_NUMERIC_CONTRACTS.json',OUT/'DEV40_512_AUDIT.json',
                 OUT/'recorder_AB/AB_RECORDER_CHECK.json',OUT/'GRIPPER_DECODER_BUG.json',
                 OUT/'S512_EXPOSURE_REPLAY.json',R85/'dita/gpu_gate4/FRESH_RELOAD_cuda.json'):
        evidence[str(path)]={'sha256':sha256_file(path),'receipt':read(path)}
    if not read(OUT/'FULL_NUMERIC_CONTRACTS.json')['passed']:raise ValueError('Native numeric parity failed')
    if not read(OUT/'recorder_AB/AB_RECORDER_CHECK.json')['passed']:raise ValueError('Actual recorder A/B not equivalent')
    if not read(R85/'dita/gpu_gate4/FRESH_RELOAD_cuda.json')['passed']:raise ValueError('Dita native reload not qualified')
    if not read(OUT/'S512_EXPOSURE_REPLAY.json')['passed_actual_stage_count_match']:raise ValueError('Exposure is not actual matched stream')
    prior=read(R85/'owner_status.json');expected=identity(prior['pid'])
    if not alive(prior['pid']) or 'VLA.recovery_r85.owner' not in expected['command']:
        raise ValueError('Resolve a new exact adoption target; never guess a PID')
    code={**native_bindings(),**dita_bindings(),**bindings()};verify(code)
    atomic_json(OUT/'ADOPT_IDENTITY.json',expected)
    atomic_json(OUT/'QUALIFICATION.json',{'passed':True,'UTC':utc(),'actual_unit_regression_tests':result.testsRun,
        'code':code,'evidence':evidence,'prior_owner_identity':expected,
        'no_final800_model_promoted_by_these_implementation_tests':True,
        'no_future_arbitrary_auto_code_repairs_or_SR_guarantees_claimed':True})
