"""Native MVD ViT-S/16, not an image ViT or a renamed TSM."""
from .vendor.modeling_finetune import vit_small_patch16_224


def build_model(*, num_classes=174, training=False):
    return vit_small_patch16_224(
        num_classes=num_classes, all_frames=16, tubelet_size=2,
        use_mean_pooling=True, use_cls_token=False,
        drop_path_rate=0.1 if training else 0.0,
        init_scale=0.001, init_values=0.0)
