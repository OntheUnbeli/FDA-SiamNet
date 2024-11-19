# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn

from mmseg.models.backbones import MobileNetV2

from opencd.registry import MODELS


@MODELS.register_module()
class Siam_Mobile(MobileNetV2):
    def __init__(self,
                 interaction_cfg=(None, None, None, None),
                 **kwargs):
        super().__init__(**kwargs)
        # assert self.num_stages == len(interaction_cfg), \
        #     'The length of the `interaction_cfg` should be same as the `num_stages`.'
        # cross-correlation
        self.ccs = []
        for ia_cfg in interaction_cfg:
            if ia_cfg is None:
                ia_cfg = dict(type='TwoIdentity')
            self.ccs.append(MODELS.build(ia_cfg))
        self.ccs = nn.ModuleList(self.ccs)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)

        outs = []
        j = 0
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x1 = layer(x1)
            x2 = layer(x2)
            # x1, x2 = self.ccs[i](x1, x2)
            if i in self.out_indices:
                outs.append(torch.cat([x1, x2], dim=1))
                x1, x2 = self.ccs[j](x1, x2)
                j += j



        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)




if __name__ == '__main__':

    a = torch.randn(1, 3, 256, 256).cuda()
    b = torch.randn(1, 3, 256, 256).cuda()
    model = Siam_Mobile()
    # model = MobileNetV2()
    model.cuda()
    out = model(a, b)
    for i in range(len(out)):
        print(out[i].shape)
