"""Bind tested automation code only after executable CPU/physical admission."""
import os,unittest
from pathlib import Path
from .common import OUT,read,bindings,atomic_json,utc
from .workflow import method_code
from VLA.common.provenance import sha256_file


def main():
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in (
        'VLA.admission_r100.tests','VLA.postfit800_r98.tests','VLA.autorefinement_r97.tests'))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise ValueError('Automation/contract tests failed')
    physics=read(OUT/'ACTUAL_PHYSICS_FIXTURE.json')
    if not physics['passed']:raise ValueError('Actual physical-gripper and RGB-D fixture required')
    transport={family:read(OUT/(family+'_MAP_TRANSPORT.json')) for family in ('simulation_reference','seer','gr00t_n15','falcon_esm')}
    if len({r['combined_sha256'] for r in transport.values()})!=1 or any(len(r['cases'])!=228 for r in transport.values()):
        raise ValueError('Actual native NumPy and simulation map bytes not equal on all76 strata')
    # Register inherited plant/metric/receipt/native source now. Do not mutate
    # any of these files while a checkpoint or evaluation uses its binding.
    code=bindings()
    for family in ('seer','gr00t_n15','falcon_esm'):code.update(method_code(family))
    proof={'passed':True,'UTC':utc(),'actual_CPU_tests':result.testsRun,'code':code,
        'physical_fixture_sha256':sha256_file(OUT/'ACTUAL_PHYSICS_FIXTURE.json'),
        'native_map_cases_per_environment':228,'initial_native_GPU_and_closed_loop_gates_still_separately_required':True,
        'queue':'admit all3 -> full native fit -> DEV40 -> bounded continuation -> CONF40 -> freeze -> actual800',
        'long_fit_concurrency':'Seer only, after real batch16/full reload/physical4 and peak<=3000MiB; stop and defer on primary risk',
        'other_families_training_waits_for_R99':True,'R99_signals_or_source_edits_authorized':False}
    atomic_json(OUT/'AUTOMATION_QUALIFICATION.json',proof)
    print({'passed':True,'tests':result.testsRun,'bound_files':len(code)},flush=True)


if __name__=='__main__':main()
