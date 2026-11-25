"""Official native OpenVLA QLoRA with the complete64 feedback source view.

Own contracts/actual two-update qualification still required. No frozen18
trainer edits, stock loss/vision/action-token code remains identical.
"""
from VLA.scene64_conditioning import PROMPT_SCHEMA
from VLA.scene64_dataset import Scene64NativeDataset


def main():
    import VLA.openvla.train_privileged_prompt as native
    native.PrivilegedNativeDataset=Scene64NativeDataset
    native.PROMPT_SCHEMA=PROMPT_SCHEMA
    native.main()


if __name__ == "__main__":main()
