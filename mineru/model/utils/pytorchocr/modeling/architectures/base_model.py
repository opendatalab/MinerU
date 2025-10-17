from torch import nn

from ..backbones import build_backbone
from ..heads import build_head
from ..necks import build_neck


class BaseModel(nn.Module):
    def __init__(self, config, **kwargs):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()

        in_channels = config.get("in_channels", 3)
        model_type = config["model_type"]
        # build backbone, backbone is need for del, rec and cls
        if "Backbone" not in config or config["Backbone"] is None:
            self.use_backbone = False
        else:
            self.use_backbone = True
            config["Backbone"]["in_channels"] = in_channels
            self.backbone = build_backbone(config["Backbone"], model_type)
            in_channels = self.backbone.out_channels

        # build neck
        # for rec, neck can be cnn,rnn or reshape(None)
        # for det, neck can be FPN, BIFPN and so on.
        # for cls, neck should be none
        if "Neck" not in config or config["Neck"] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config["Neck"]["in_channels"] = in_channels
            self.neck = build_neck(config["Neck"])
            in_channels = self.neck.out_channels

        # # build head, head is need for det, rec and cls
        if "Head" not in config or config["Head"] is None:
            self.use_head = False
        else:
            self.use_head = True
            config["Head"]["in_channels"] = in_channels
            self.head = build_head(config["Head"], **kwargs)

        self.return_all_feats = config.get("return_all_feats", False)

        self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        y = dict()
        if self.use_backbone:
            x = self.backbone(x)
        if isinstance(x, dict):
            y.update(x)
        else:
            y["backbone_out"] = x
        final_name = "backbone_out"
        if self.use_neck:
            x = self.neck(x)
            if isinstance(x, dict):
                y.update(x)
            else:
                y["neck_out"] = x
            final_name = "neck_out"
        if self.use_head:
            x = self.head(x)
        # for multi head, save ctc neck out for udml
        if isinstance(x, dict) and "ctc_nect" in x.keys():
            y["neck_out"] = x["ctc_neck"]
            y["head_out"] = x
        elif isinstance(x, dict):
            y.update(x)
        else:
            y["head_out"] = x
        if self.return_all_feats:
            if self.training:
                return y
            elif isinstance(x, dict):
                return x
            else:
                return {final_name: x}
        else:
            return x
