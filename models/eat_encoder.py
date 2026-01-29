import torch
import timm


def load(name, checkpoint_path=None, in_chans=1):
    if name != "eat_base_10ep":
        raise ValueError("Unsupported EATModel name. Use 'eat_base_10ep'.")
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,
        in_chans=in_chans,
    )
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
    return model
