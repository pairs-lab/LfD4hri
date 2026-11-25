import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from VLA.audit_finetune_data import save_contact_sheets


class ContactSheetSubsetTest(unittest.TestCase):
    def test_explicit_subset_changes_only_export_dimensions(self):
        arrays = {"canonical_actions": np.zeros((2, 7)),
                  "front_rgb": np.zeros((2, 4, 4, 3), np.uint8),
                  "wrist_rgb": np.zeros((2, 4, 4, 3), np.uint8)}
        context = unittest.mock.MagicMock()
        context.__enter__.return_value = arrays
        with tempfile.TemporaryDirectory() as directory, patch("VLA.audit_finetune_data.np.load", return_value=context):
            output = Path(directory)
            save_contact_sheets(output, {"move": [Path("fake")]}, output, task_names=("move",))
            with Image.open(output/"camera_contact_sheet.png") as sheet:
                self.assertEqual(sheet.size, (1536, 256))
            with self.assertRaises(ValueError):
                save_contact_sheets(output, {"move": [Path("fake")]}, output)
