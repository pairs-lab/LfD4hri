from types import SimpleNamespace
import unittest
from VLA.pi0.augmentation import colour_safe_factory, install_colour_safe_augmentation


class FakeJitter:
    def __init__(self, brightness=.1, contrast=.1, saturation=.1, hue=.1, p=.5):
        self.settings = (brightness, contrast, saturation, hue, p)


class ColourSafeAugmentationTest(unittest.TestCase):
    def test_only_hue_is_disabled(self):
        instance = colour_safe_factory(FakeJitter)(brightness=.3, contrast=.4, saturation=.5)
        self.assertEqual(instance.settings, (.3, .4, .5, 0, .5))

    def test_positional_hue_and_install_are_safe(self):
        module = SimpleNamespace(ColorJitter=FakeJitter)
        install_colour_safe_augmentation(module)
        constructor = module.ColorJitter
        install_colour_safe_augmentation(module)
        self.assertIs(module.ColorJitter, constructor)
        self.assertEqual(module.ColorJitter(.3, .4, .5, .2, 1).settings, (.3, .4, .5, 0, 1))


if __name__ == "__main__":
    unittest.main()
