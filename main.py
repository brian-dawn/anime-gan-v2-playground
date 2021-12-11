import torch
import numpy as np
import cv2
from PIL import Image
from torch._C import device

pretrained_model_name = "face_paint_512_v2"  # or paprika, celeba_distill, or face_paint_512_v1, or face_paint_512_v2
model = torch.hub.load(
    "bryandlee/animegan2-pytorch:main", "generator", pretrained=pretrained_model_name
)
model = model.to("cuda")
face2paint = torch.hub.load(
    "bryandlee/animegan2-pytorch:main", "face2paint", size=512, device="cuda"
)

# define a video capture object
vid = cv2.VideoCapture(0)

while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    out = face2paint(model, im_pil)

    updated_frame = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    cv2.imshow("frame", updated_frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
