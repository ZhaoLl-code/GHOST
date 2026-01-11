import torch
import cv2
import numpy as np
from TransDINO.TransDINO import TransDINO, dino_model, n_layers

def main(data_path, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransDINO().to(device)
    model.load_state_dict(torch.load(r"checkpoints\transdino.pth", map_location=device))
    model.eval()

    img = cv2.imread(data_path, cv2.IMREAD_COLOR).astype(np.float32)
    H, W = img.shape[:2]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device) # 1 * 3 * H * W
    img = model.preprocess_image(img, (640, 480))
    feats = dino_model.get_intermediate_layers(img, n=range(n_layers), reshape=True, norm=True)
    feats = feats[-1]
    logit = model(img, feats)
    logit = model.postprocess_mask(logit, (H, W))
    logit = logit.squeeze(0).squeeze(0)
    pred_mask = (torch.sigmoid(logit) > 0.5).cpu().detach().numpy().astype(np.uint8) * 255
    cv2.imwrite(save_path, pred_mask)


if __name__ == "__main__":
    data_path = r"images/rgb.jpg"
    save_path = r'mask.png'
    main(data_path, save_path)