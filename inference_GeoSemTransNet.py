import pathlib
import cv2
import numpy as np
import torch
import OpenEXR
import Imath
from GeoSemTransNet.GeoSemTransNet import GeoSemTransNet
from TransDINO.TransDINO import dino_model, n_layers
from TransDecomp.TransDecomp import TransDecomp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dino_model = dino_model.to(device)
dino_model.eval()

def read_exr(exr_path):
    exr_file = OpenEXR.InputFile(exr_path)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    allchannels = []
    for c in ['R', 'G', 'B']:
        channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
        channel.shape = (size[1], size[0])
        allchannels.append(channel)
    exr_arr = np.array(allchannels)
    print('Exr shape:', exr_arr.shape, exr_arr.dtype, exr_arr.min(), exr_arr.max())
    rgb_arr = exr_arr * -1
    rgb_arr[0, :, :] *= -1
    rgb_arr = (((rgb_arr + 1) / 2.0) * 255).astype(np.uint8).transpose(1, 2, 0)
    cv2.imwrite("normal_translated.jpg", rgb_arr)
    return exr_arr

def read_normal_map(image_path):
    img_array = cv2.imread(image_path).astype(np.float32)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    if img_array.shape[-1] > 3:
        img_array = img_array[:, :, :3]

    normal_vectors = (img_array / 255.0) * 2.0 - 1.0

    norm = np.linalg.norm(normal_vectors, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    normal_vectors = normal_vectors / norm

    return normal_vectors.transpose((2, 0, 1))

def normalize_normals(normals, eps=1e-8):
    magnitude = torch.norm(normals, dim=1, keepdim=True)
    magnitude = torch.clamp(magnitude, min=eps)
    normalized_normals = normals / magnitude

    return normalized_normals

def resize_to_512x512(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        out = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    elif img.ndim == 3 and img.shape[2] == 3:
        out = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    return out

def is_unit_normal(normal: torch.Tensor, eps=1e-4):
    n = normal.moveaxis(1, -1)          # (B,H,W,3)
    norm = n.norm(dim=-1)               # (B,H,W)
    max_err = (norm - 1.0).abs().max()
    return max_err.item() < eps

def inference(rgb_path, mask_path, normal_path, generated_rgb_path):
    decompose_model = TransDecomp().to(device)
    decompose_model.eval()
    decompose_model.load_state_dict(torch.load(pathlib.Path(r'checkpoints\transdecomp.pth')))

    # 使用方法:
    geoSemTransNet = GeoSemTransNet().to(device)
    geoSemTransNet.eval()
    geoSemTransNet.load_state_dict(torch.load(r"checkpoints\geosemtransnet.pth"))

    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if normal_path.endswith('.exr'):
        normal = read_exr(normal_path)
        normal = normal.transpose((1, 2, 0))
        normal *= -1
        normal[:, :, 0] *= -1
    else:
        normal = read_normal_map(normal_path)
        normal = normal.transpose((1, 2, 0))

    source_H, source_W = bgr.shape[:2]

    rgb = resize_to_512x512(rgb) / 255
    mask = resize_to_512x512(mask)
    normal = resize_to_512x512(normal) / 255

    rgb = rgb.transpose((2, 0, 1))
    mask = (mask > 0).astype(np.float32)
    mask = np.expand_dims(mask, axis=0)
    normal = normal.transpose((2, 0, 1))

    rgb = torch.from_numpy(rgb).float().cuda().unsqueeze(0)
    mask = torch.from_numpy(mask).float().cuda().unsqueeze(0)
    normal = torch.from_numpy(normal).float().cuda().unsqueeze(0)
    normal = normalize_normals(normal)

    print(rgb.shape, mask.shape)
    with torch.no_grad():
        pred_alpha, pred_fg = decompose_model(rgb, mask)
        feats = dino_model.get_intermediate_layers(rgb, n=range(n_layers), reshape=True, norm=True)
        feats = feats[-1]
        print(feats.shape)

    generated_rgb = geoSemTransNet(rgb, mask, normal, pred_alpha, pred_fg, feats)
    generated_rgb = generated_rgb.detach().cpu().numpy().squeeze(0)
    generated_rgb = generated_rgb.transpose(1, 2, 0)
    generated_rgb = generated_rgb * 255
    generated_rgb = cv2.resize(generated_rgb, (1920, 1080))
    generated_rgb = generated_rgb.astype(np.uint8)
    generated_rgb = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(generated_rgb_path, generated_rgb)

if __name__ == '__main__':
    rgb_path = r'images\rgb.jpg'
    mask_path = r'images\mask.png'
    normal_path = r'F:\GHOST\normal_translated.jpg'
    generated_rgb_path = r"generated_rgb.jpg"
    inference(rgb_path, mask_path, normal_path, generated_rgb_path)
