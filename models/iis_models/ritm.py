import inspect
import torch
import torch.nn as nn

from models.backbones.hrnet_ocr.hrnet_ocr import HighResolutionNet


def get_class_from_str(class_str):
    components = class_str.split('.')
    mod = __import__('.'.join(components[:-1]))
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class LRMult(object):
    def __init__(self, lr_mult=1.):
        self.lr_mult = lr_mult

    def __call__(self, m):
        if getattr(m, 'weight', None) is not None:
            m.weight.lr_mult = self.lr_mult
        if getattr(m, 'bias', None) is not None:
            m.bias.lr_mult = self.lr_mult


class ModTrain(object):
    def __init__(self, mode):
        self.mode = mode
    
    def __call__(self, m):
        if not self.mode or not isinstance(m, nn.BatchNorm2d):
            m.training = self.mode


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
        )

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale


class HRNetISModel(nn.Module):

    def __init__(self, use_leaky_relu=False,
                 width=48, ocr_width=256, small=False, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.feature_extractor = HighResolutionNet(width=width, ocr_width=ocr_width, small=small,
                                                   num_classes=1, norm_layer=norm_layer)
        self.feature_extractor.apply(LRMult(backbone_lr_mult))
        if ocr_width > 0:
            self.feature_extractor.ocr_distri_head.apply(LRMult(1.0))
            self.feature_extractor.ocr_gather_head.apply(LRMult(1.0))
            self.feature_extractor.conv3x3_ocr.apply(LRMult(1.0))

        self.coord_feature_ch = 2

        mt_layers = [
            nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
            ScaleLayer(init_value=0.05, lr_mult=1)
        ]
        self.maps_transform = nn.Sequential(*mt_layers)


    def train(self, mode=True):
        self.training = mode
        self.apply(ModTrain(mode))
        return self


    def forward(self, image, coord_features):
        coord_features = self.maps_transform(coord_features)
        outputs = self.backbone_forward(image, coord_features)

        outputs = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        return outputs


    def backbone_forward(self, image, coord_features=None):
        net_outputs = self.feature_extractor(image, coord_features)

        return {'instances': net_outputs[0], 'instances_aux': net_outputs[1]}


    @classmethod
    def default_params(cls):
        params = dict()
        for mclass in HRNetISModel.mro():
            if mclass is nn.Module or mclass is object:
                continue

        mclass_params = inspect.signature(HRNetISModel.__init__).parameters
        for pname, param in mclass_params.items():
            if param.default != param.empty and pname not in params:
                params[pname] = param

        return params


    @classmethod
    def load_from_checkpoint(cls, path, verbose=False, **kwargs):
        ckpt = torch.load(path, map_location='cpu')
        config = ckpt['config']
        state_dict = ckpt['state_dict']

        default_params = cls.default_params()
        model_args = dict()
        for pname, param in config['params'].items():
            value = param['value']
            if param['type'] == 'class':
                value = get_class_from_str(value)

            if pname not in default_params:
                if verbose:
                    print('unmatched key:', pname, param)
                continue

            if not param['specified'] and default_params[pname].default == value:
                continue
            model_args[pname] = value

        model_args.update(kwargs)
        model = cls(**model_args)
        model.load_state_dict(state_dict, strict=False)
        return model

    def save_checkpoint(self, in_path, out_path):
        ckpt = torch.load(in_path, map_location='cpu')
        config = ckpt['config']
        ckpt['state_dict'] = self.state_dict()
        torch.save(ckpt, out_path)
