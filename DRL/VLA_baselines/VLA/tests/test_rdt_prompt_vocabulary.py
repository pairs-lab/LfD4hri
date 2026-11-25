import unittest
from config import INSTRUCTION_FORM, PICK_TARGETS, PLACE_TARGETS
from VLA.rdt.prompt_vocabulary import declared_prompt_vocabulary, vocabulary_sha256


class RDTPromptVocabularyTest(unittest.TestCase):
    def test_all_valid_language_combinations_without_scene_data(self):
        prompts = declared_prompt_vocabulary()
        expected = 2 * len(PICK_TARGETS) + 2 * (len(PICK_TARGETS) * len(PLACE_TARGETS) - len(set(PICK_TARGETS) & set(PLACE_TARGETS)))
        self.assertEqual(len(prompts), expected)
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertIn(INSTRUCTION_FORM[2].format("red apple", "brown bowl"), prompts)
        self.assertNotIn(INSTRUCTION_FORM[2].format("blue block", "blue block"), prompts)
        self.assertEqual(len(vocabulary_sha256()), 64)


if __name__ == "__main__":
    unittest.main()
