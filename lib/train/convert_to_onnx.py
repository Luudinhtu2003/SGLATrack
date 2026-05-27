import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.sglatrack import build_sglatrack
# forward propagation related
from lib.train.actors import sglatrackActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


class ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, template, search):
        out = self.model.forward_test(template, search)

        # chọn tensor output cần export
        if isinstance(out, dict):
            return out["pred_boxes"]

        return out


def run(settings):

    # =========================
    # Load config
    # =========================
    if not os.path.exists(settings.cfg_file):
        raise ValueError(f"{settings.cfg_file} doesn't exist.")

    config_module = importlib.import_module(
        f"lib.config.{settings.script_name}.config"
    )

    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)

    update_settings(settings, cfg)

    # =========================
    # Build model
    # =========================
    if settings.script_name == "sglatrack":
        net = build_sglatrack(cfg)
    else:
        raise ValueError("illegal script name")

    # =========================
    # Load checkpoint
    # =========================
    checkpoint_path = r"/home/getac2/tuld3/tu_workspace/track_uav/SGLATrack/results/train/sglatrackdeit_distilled/sglatrack_ep0297.pth.tar"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "net" in checkpoint:
        net.load_state_dict(checkpoint["net"], strict=False)
    else:
        net.load_state_dict(checkpoint, strict=False)

    net.eval()
    net.cuda()

    # =========================
    # Wrapper
    # =========================
    wrapper = ONNXWrapper(net).cuda().eval()

    # =========================
    # Dummy input
    # =========================
    template_size = cfg.TEST.TEMPLATE_SIZE
    search_size = cfg.TEST.SEARCH_SIZE

    template = torch.randn(
        1, 3, template_size, template_size
    ).cuda()

    search = torch.randn(
        1, 3, search_size, search_size
    ).cuda()

    # =========================
    # Export ONNX
    # =========================
    output_path = "F:\Tu_workspace\SGLATrack\onnx_models\sgla_op11_v5.onnx"

    torch.onnx.export(
        wrapper,
        (template, search),
        output_path,
        opset_version=11,
        input_names=["template", "search"],
        output_names=["pred_boxes"],
    )

    print(f"ONNX model exported to: {output_path}")
