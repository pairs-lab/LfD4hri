"""Regression qualification; zero new neural robot episodes in this module."""
import unittest
from .common import OUT, ROOT, PARENT, R104, bindings, read, atomic_json, verify, utc
from VLA.common.provenance import sha256_file


def main():
    from VLA.admission_r100.workflow import method_code
    # Verify original sources against the immutable already-certified R104
    # method BEFORE binding this new scheduler's sources.
    original=read(R104/'falcon_esm/FROZEN_METHOD.json')
    verify(original['code'])
    suites=('VLA.admission_r100.tests','VLA.postfit800_r98.tests',
        'VLA.autorefinement_r97.tests','VLA.continuation_review_r101_tests',
        'VLA.falcon_esm_r91.tests','VLA.direct800_r104.tests','VLA.falcon_final_r105.tests')
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in suites)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise ValueError('Final continuation regressions failed')
    verify(original['code'])
    code={**original['code'],**method_code('falcon_esm'),**bindings()}
    verify(code)
    atomic_json(OUT/'AUTOMATION_QUALIFICATION.json',{'passed':True,'UTC':utc(),
        'CPU_tests_passed':result.testsRun,'code':code,
        'PLAN_sha256':sha256_file(ROOT/'VLA/falcon_final_r105/PLAN.md'),
        'original_frozen_sources_unchanged':True,'new_robot_trials_in_CPU_tests':0,
        'parent_checkpoint':str(PARENT),
        'no_original_code_weights_physics_or_threshold_files_modified':True})
    print({'passed':True,'CPU_tests':result.testsRun,'bound_source_files':len(code)},flush=True)


if __name__=='__main__':main()
