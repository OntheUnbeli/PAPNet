from .metrics import averageMeter, runningScore
from .log import get_logger
from .loss import MscCrossEntropyLoss
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, adjust_lr
from .ranger.ranger import Ranger
from .ranger.ranger913A import RangerVA
from .ranger.rangerqh import RangerQH

def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'sunrgbd224', 'SUIM']

    if cfg['dataset'] == 'nyuv2':
        from .datasets.suim import NYUv2
        return NYUv2(cfg, mode='train'), NYUv2(cfg, mode='test')
    if cfg['dataset'] == 'sunrgbd':
        from .datasets.sunrgbd import SUNRGBD
        return SUNRGBD(cfg, mode='train'), SUNRGBD(cfg, mode='test')
    if cfg['dataset'] == 'sunrgbd224':
        from .datasets.sunrgbd224 import SUNRGBD224
        return SUNRGBD224(cfg, mode='train'), SUNRGBD224(cfg, mode='test')
    if cfg['dataset'] == 'SUIM':
        from .datasets.suim import SUIM
        return SUIM(cfg, mode='train'), SUIM(cfg, mode='test')


def get_model(cfg):
    # if cfg['model_name'] == 'DGPI':
    #     from .model.DGPINet_T import EnDecoderModel
    #     return EnDecoderModel(n_classes=8, backbone='segb2')

    # if cfg['model_name'] == 'fake':
    #     from .model.Mine.MAIN.Fakenet_seg import FakeNet
    #     return FakeNet()
    if cfg['model_name'] == 'SAM':
        from Uw_SAM_pro import EnDecoderModel
        return EnDecoderModel(num_classes=8)
    if cfg['model_name'] == 'Sam':
        from Uw_SAM import EnDecoderModel
        return EnDecoderModel(num_classes=8)



def get_mutual_model(cfg):
    if cfg['model_name'] == 'fakev1':
        from .model.Mine_New.Net1.TrueNetv1 import EnDecoderModel
        return EnDecoderModel(n_classes=8, backbone='convnext_tiny')


