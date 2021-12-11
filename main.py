# pip install mss pyvirtualcam facenet_pytorch pykalman

import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw

from mss import mss
import pyvirtualcam

from pykalman import KalmanFilter

# Local libs.
import detection
import anime_gan
import facetracking


# define a video capture object
vid = cv2.VideoCapture(0)

face_observations = []

kf = None
filtered_state_means = None
filtered_state_covariances = None

virtual_cam = None
while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # with mss() as sct:
    #     monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
    #     frame = np.array(sct.grab(monitor))
    #     # Remove the alpha channel.
    #     frame = frame[:, :, :3]
    boxes = facetracking.detect_faces(frame)

    # No faces detected.
    if boxes is not None and len(boxes) != 0:

        round_to = 1
        [x1, y1, x2, y2] = [
            int(round(x / round_to) * round_to) for x in boxes[0].tolist()
        ]

        percent = 0.5

        width = x2 - x1
        height = y2 - y1
        x1 = max(0, x1 - width * percent)
        x2 = min(frame.shape[1], x2 + width * percent)
        y1 = max(0, y1 - height * percent)
        y2 = min(frame.shape[0], y2 + height * percent)
        coords = [x1, y1, x2, y2]

        face_observations.append(coords)

    # If we have a face go to that, otherwise display the entire image.
    if len(face_observations) != 0:
        obs_to_average = 5
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

    out = anime_gan.anime(frame)
    updated_frame = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)

    height, width, layers = updated_frame.shape
    updated_frame = cv2.resize(updated_frame, (height * 2, width * 2))

    if virtual_cam is None:
        print("Make sure you're running OBS!")

        height, width, layers = updated_frame.shape
        virtual_cam = pyvirtualcam.Camera(width=width, height=height, fps=30)
        print(f"Using virtual camera: {virtual_cam.device}")

    # frame[:] = cam.frames_sent % 255  # grayscale animation

    updated_frame = cv2.cvtColor(updated_frame, cv2.COLOR_BGR2RGB)
    virtual_cam.send(updated_frame)
    virtual_cam.sleep_until_next_frame()

    # cv2.imshow("frame", updated_frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

virtual_cam.close()
vid.release()
cv2.destroyAllWindows()
