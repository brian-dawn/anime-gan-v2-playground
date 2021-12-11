import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw

# Stuff for face tracking/cropping.
from facenet_pytorch import InceptionResnetV1, MTCNN
from pykalman import KalmanFilter

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

face_observations = []

kf = None
filtered_state_means = None
filtered_state_covariances = None
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

    percent = 0.5

    width = x2 - x1
    height = y2 - y1
    x1 = max(0, x1 - width * percent)
    x2 = min(frame.shape[1], x2 + width * percent)
    y1 = max(0, y1 - height * percent)
    y2 = min(frame.shape[0], y2 + height * percent)
    coords = [x1, y1, x2, y2]

    obs_to_average = 10
    face_observations.append(coords)
    face_observations = face_observations[-obs_to_average:]
    [x1, y1, x2, y2] = [int(x) for x in np.average(face_observations, axis=0)]
    # TODO: attempt to get kalman filters working.
    # if kf is None:
    #     kf = KalmanFilter(initial_state_mean=coords)
    #     filtered_state_means = kf.initial_state_mean
    #     filtered_state_covariances = kf.initial_state_covariance
    #     filtered_state_means, filtered_state_covariances = kf.em(coords).smooth(coords)
    #     print("Initialized Kalman Filter")
    #     print(filtered_state_means)
    #     print(filtered_state_covariances)
    # filtered_state_means, filtered_state_covariances = kf.filter_update(
    #     filtered_state_means[-1], filtered_state_covariances[-1], observation=coords
    # )
    # # smoothed = kf.em(coords).smooth(coords)[0][:, 0]
    # smoothed = filtered_state_means[0][:, 0]
    # [x1, y1, x2, y2] = [int(x) for x in smoothed]
    # print(x1, y1, x2, y2)

    # Crop it.
    frame = frame[y1:y2, x1:x2]

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    out = face2paint(model, im_pil)

    updated_frame = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)

    height, width, layers = updated_frame.shape
    updated_frame = cv2.resize(updated_frame, (height * 2, width * 2))
    cv2.imshow("frame", updated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
