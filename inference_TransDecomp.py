import cv2
import numpy as np
import torch
from TransDecomp.TransDecomp import TransDecomp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def resize_to_256x256(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        out = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    elif img.ndim == 3 and img.shape[2] == 3:
        out = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    return out
def inference(rgb_path, mask_path, afg_path):
    model = TransDecomp().to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/transdecomp.pth"))
    model.eval()

    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    H, W = rgb.shape[:2]
    rgb = resize_to_256x256(rgb) / 255
    mask = resize_to_256x256(mask)

    mask = (mask > 0).astype(np.float32)
    input_rgb = rgb.transpose((2, 0, 1))
    input_mask = np.expand_dims(mask, axis=0)

    with torch.no_grad():
        input_rgb = torch.from_numpy(input_rgb).unsqueeze(0).float().to(DEVICE)
        input_mask = torch.from_numpy(input_mask).unsqueeze(0).float().to(DEVICE)

        pred_alpha, pred_fg = model(input_rgb, input_mask)
        alpha = pred_alpha.squeeze(0).squeeze(0).cpu().numpy()

        pred_fg_3ch = pred_fg.repeat(1, 3, 1, 1)
        fg = pred_fg_3ch.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        new_rgb_fg = rgb
        alpha = alpha[..., np.newaxis]
        mask = mask[..., np.newaxis]
        blended = fg * alpha
        new_rgb_fg = np.where(mask > 0, blended, new_rgb_fg)

        new_rgb_fg *= 255
        new_rgb_fg = new_rgb_fg.astype(np.uint8)
        new_rgb_fg = cv2.cvtColor(new_rgb_fg, cv2.COLOR_RGB2BGR)

        new_rgb_fg = cv2.resize(new_rgb_fg, (W, H), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(afg_path, new_rgb_fg)

if __name__ == "__main__":
    rgb_path = r"images/rgb.jpg"
    mask_path = r'images/mask.png'
    afg_path = r"afg.jpg"
    inference(rgb_path, mask_path, afg_path)