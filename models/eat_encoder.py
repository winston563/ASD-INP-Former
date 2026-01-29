import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _resize_pos_embed(pos_embed, new_token_count):
    if pos_embed.shape[1] == new_token_count:
        return pos_embed
    orig_tokens = pos_embed.shape[1]
    orig_grid_size = int(math.sqrt(orig_tokens - 1))
    num_prefix = orig_tokens - orig_grid_size * orig_grid_size
    new_grid_tokens = new_token_count - num_prefix
    new_grid_size = int(math.sqrt(new_grid_tokens)) if new_grid_tokens > 0 else 0
    if new_grid_size * new_grid_size != new_grid_tokens:
        return pos_embed[:, :new_token_count]
    prefix = pos_embed[:, :num_prefix]
    pos_tokens = pos_embed[:, num_prefix:]
    pos_tokens = pos_tokens.reshape(1, orig_grid_size, orig_grid_size, -1).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=(new_grid_size, new_grid_size), mode="bilinear", align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, -1)
    return torch.cat([prefix, pos_tokens], dim=1)


class EATEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.blocks = model.blocks
        self.num_register_tokens = 0

    def prepare_tokens(self, x):
        x = self.model.patch_embed(x)
        if hasattr(self.model, "_pos_embed"):
            x = self.model._pos_embed(x)
        else:
            if self.model.cls_token is not None:
                cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_token, x), dim=1)
            if self.model.pos_embed is not None:
                pos_embed = _resize_pos_embed(self.model.pos_embed, x.shape[1])
                x = x + pos_embed
            x = self.model.pos_drop(x)
        if hasattr(self.model, "norm_pre") and self.model.norm_pre is not None:
            x = self.model.norm_pre(x)
        return x

    def forward(self, x):
        return self.model(x)


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
    return EATEncoderWrapper(model)
