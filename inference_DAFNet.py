import json
import pathlib
import cv2
import numpy as np
import torch
from TransDINO.TransDINO import dino_model, n_layers
from TransDecomp.TransDecomp import TransDecomp
from DAFNet.DAFNet import DAFNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dino_model.eval()

def resize_to_512x512(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        out = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    elif img.ndim == 3 and img.shape[2] == 3:
        out = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    return out

def is_unit_normal(normal: torch.Tensor, eps=1e-4):
    n = normal.moveaxis(1, -1)
    norm = n.norm(dim=-1)
    max_err = (norm - 1.0).abs().max()
    return max_err.item() < eps

def inference(rgb_path, mask_path, normal_path):
    decompose_model = TransDecomp().to(device)
    decompose_model.eval()
    decompose_model.load_state_dict(torch.load(pathlib.Path(r'checkpoints\transdecomp.pth')))

    normalpredictor_model = DAFNet().cuda()
    normalpredictor_model.eval()
    normalpredictor_model.load_state_dict(torch.load(pathlib.Path(r'checkpoints\dafnet.pth')))

    with torch.no_grad():
        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        source_H, source_W = bgr.shape[:2]

        rgb = resize_to_512x512(rgb) / 255
        mask = resize_to_512x512(mask)

        mask = (mask > 0).astype(np.float32)
        rgb = rgb.transpose((2, 0, 1))

        rgb = torch.from_numpy(rgb).float().cuda().unsqueeze(0)
        mask = torch.from_numpy(mask).float().cuda().unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred_alpha, pred_fg = decompose_model(rgb, mask)
            feats = dino_model.get_intermediate_layers(rgb, n=range(n_layers), reshape=True, norm=True)
            feats = feats[-1]

        output_dict = normalpredictor_model(rgb, mask, pred_alpha, pred_fg, feats)

        pred_norm = output_dict["pred_norm"]
        pred_kappa = output_dict["pred_kappa"]

        pred_norm *= -1
        pred_norm[:, :, :, 0] *= -1
        pred_norm = (pred_norm + 1) / 2
        pred_norm = pred_norm * mask + (1 - mask) * rgb
        pred_norm = pred_norm.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        pred_norm *= 255
        pred_norm = pred_norm.astype(np.uint8)
        pred_norm = cv2.cvtColor(pred_norm, cv2.COLOR_RGB2BGR)
        pred_norm = cv2.resize(pred_norm, (source_W, source_H), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(normal_path, pred_norm)

if __name__ == '__main__':
    rgb_path = r"images/rgb.jpg"
    mask_path = r"images/mask.png"
    normal_path = r"normal.jpg"
    inference(rgb_path, mask_path, normal_path)
