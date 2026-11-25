from types import SimpleNamespace
import unittest
import numpy as np
from VLA.pi0.fast_decoder_integrity import require_actual_coefficients


class Tests(unittest.TestCase):
    def processor(self,count):return SimpleNamespace(min_token=-1,bpe_tokenizer=SimpleNamespace(
        get_vocab=lambda:{'a':0},decode=lambda tokens:chr(1)*count))
    def test_malformed_count_is_not_a_fake_zero_action(self):
        with self.assertRaisesRegex(ValueError,'1024 DCT coefficients, expected 35'):
            require_actual_coefficients(self.processor(1024),[0],5,7)
    def test_valid_neural_zero_action_is_allowed(self):
        actual=require_actual_coefficients(self.processor(35),[0],5,7)
        self.assertTrue(np.all(actual==0));self.assertEqual(actual.shape,(5,7))
    def test_original_FAST_native_layout_is_valid(self):
        self.assertEqual(require_actual_coefficients(self.processor(1024),[0],32,32).shape,(32,32))
    def test_unknown_BPE_id_rejected(self):
        with self.assertRaisesRegex(ValueError,'BPE IDs'):require_actual_coefficients(self.processor(35),[1],5,7)


if __name__=='__main__':unittest.main()
