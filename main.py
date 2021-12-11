import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw

from facenet_pytorch import InceptionResnetV1, MTCNN

pretrained_model_name = "face_paint_512_v2"  # or paprika, celeba_distill, or face_paint_512_v1, or face_paint_512_v2
model = torch.hub.load(
    "bryandlee/animegan2-pytorch:main", "generator", pretrained=pretrained_model_name
)
model = model.to("cuda")
face2paint = torch.hub.load(
    "bryandlee/animegan2-pytorch:main", "face2paint", size=512, device="cuda"
)

# face tracking
mtcnn = MTCNN(keep_all=True, device="cuda")

# define a video capture object
vid = cv2.VideoCapture(0)

while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Detect faces
    boxes, _ = mtcnn.detect(frame)

    # No faces detected.
    if boxes is None or len(boxes) == 0:
        continue

    round_to = 1
    [x1, y1, x2, y2] = [int(round(x / round_to) * round_to) for x in boxes[0].tolist()]
    pad = 50
    x1 = max(0, x1 - pad)
    x2 = min(frame.shape[1], x2 + pad)
    y1 = max(0, y1 - pad)
    y2 = min(frame.shape[0], y2 + pad)
    frame = frame[y1:y2, x1:x2]

    # draw = ImageDraw.Draw(frame)
    # for box in boxes:
    #    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

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
