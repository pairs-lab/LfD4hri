import ast
from pathlib import Path
import unittest


class OpenVLAQualificationDispatchTests(unittest.TestCase):
    def test_training_declares_model_and_actual_optimizer_budget(self):
        path = Path(__file__).resolve().parents[1] / "openvla/finetune.py"
        tree = ast.parse(path.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "require_training_source"]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        self.assertEqual(ast.literal_eval(keywords["model"]), "openvla")
        self.assertEqual(ast.unparse(keywords["optimizer_updates"]), "args.steps")


if __name__ == "__main__":
    unittest.main()
