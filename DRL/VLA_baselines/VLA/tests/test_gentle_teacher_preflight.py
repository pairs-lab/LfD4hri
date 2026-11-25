from pathlib import Path
import unittest
from unittest.mock import patch
from VLA.preflight_gentle_teacher import select_cases,TASKS


class GentleTeacherPreflightTest(unittest.TestCase):
    def inputs(self):
        source=str(Path("/tmp/teacher_preflight").resolve())
        index=[{"file":f"{task}_{name}","task":{"name":task,"pick_name":name}}
               for task in TASKS for name in ("first","second")]
        declaration=[{"source":source,"file":r["file"],"sha256":"abc"} for r in index]
        return source,index,declaration

    @patch("VLA.preflight_gentle_teacher.PICK_TARGETS",{"first":{},"second":{}})
    def test_full_vocabulary_no_validation_substitution(self):
        source,index,declaration=self.inputs()
        self.assertEqual(len(select_cases(index,declaration,source)),8)
        with self.assertRaises(ValueError):
            select_cases(index,declaration[:-1],source)

    @patch("VLA.preflight_gentle_teacher.PICK_TARGETS",{"first":{},"second":{}})
    def test_deterministic_first_case_and_no_paired_or_recovery_selection(self):
        source,index,declaration=self.inputs()
        duplicate={**index[0],"file":"extra"}
        declaration.append({"source":source,"file":"extra","sha256":"abc"})
        selected=select_cases(index+[duplicate],declaration,source)
        self.assertEqual(selected[0][0]["file"],index[0]["file"])
        for marker in ({"paired_scene_source":"old"},{"recovery_start_actions_not_training_labels":[[0]*7]}):
            changed=[{**r,**marker} if r==index[-1] else r for r in index]
            with self.assertRaises(ValueError):
                select_cases(changed,declaration,source)
