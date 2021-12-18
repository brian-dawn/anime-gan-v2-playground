import cv2
import numpy as np
import torch

from PIL import Image, ImageDraw

pretrained_model_name = "face_paint_512_v2"  # or paprika, celeba_distill, or face_paint_512_v1, or face_paint_512_v2
model = torch.hub.load(
    "bryandlee/animegan2-pytorch:main", "generator", pretrained=pretrained_model_name
)
model = model.to("cuda")
face2paint = torch.hub.load(
    "bryandlee/animegan2-pytorch:main", "face2paint", size=512, device="cuda"
)


def anime(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    out = face2paint(model, im_pil)

    updated_frame = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    return updated_frame
