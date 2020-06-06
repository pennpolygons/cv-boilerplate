import torchvision.models as models
import torch.nn as nn

from omegaconf import DictConfig

import pdb

# FIXME:
def get_network(cfg: DictConfig) -> nn.Module:
    if cfg.backbone_name == "vgg19":
        return models.vgg19(pretrained=True, progress=True)
    else:
        return models.resnet50(pretrained=True, progress=True)
    pass
