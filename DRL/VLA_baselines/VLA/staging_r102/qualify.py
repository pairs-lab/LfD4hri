"""Executable regression evidence without rebinding the live R100 owner."""
import unittest
from .common import OUT,R100,ROOT,read,utc,atomic_json
from VLA.admission_r100.common import verify
from VLA.common.provenance import sha256_file


def main():
    q=read(R100/'AUTOMATION_QUALIFICATION.json');verify(q['code'])
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in (
        'VLA.staging_r102.tests','VLA.staging_r102.scan_tests','VLA.admission_r100.tests','VLA.postfit800_r98.tests',
        'VLA.autorefinement_r97.tests','VLA.continuation_review_r101_tests'))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise ValueError('Staging regression failed')
    atomic_json(OUT/'VALIDATION.json',{'passed':True,'UTC':utc(),'tests_passed':result.testsRun,
        'existing_source_files_unchanged':len(q['code']),
        'source':{str(p):sha256_file(p) for p in (ROOT/'VLA/staging_r102').glob('*.py')},
        'all_tests_are_synthetic_NOT_robot_success_trials':True,'live_R100_qualification_NOT_overwritten':True})


if __name__=='__main__':main()
