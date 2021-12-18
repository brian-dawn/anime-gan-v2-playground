# pip install mss pyvirtualcam facenet_pytorch pykalman
import io
import signal
import time

import torch
import numpy as np
import cv2
import os
import sys


from PIL import Image, ImageDraw

from mss import mss
import pyvirtualcam

from pykalman import KalmanFilter

import rembg.bg as bg

# Local libs.
import detection
import anime_gan
import facetracking


# define a video capture object
vid = cv2.VideoCapture(0)


# frame = cv2.imread("c:/Users/Brian/Downloads/wallace.jpg")
# if frame is None:
#     print("nope")
#     sys.exit(1)
# else:
#     print("yep")


def track_faces(frame, face_observations):

    boxes = facetracking.detect_faces(frame)

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


def crop_face(frame):
    global face_observations

    try:
        face_observations
    except NameError:
        face_observations = []

    track_faces(frame, face_observations)

    # If we have a face go to that, otherwise display the entire image.
    if len(face_observations) != 0:
        obs_to_average = 5
        face_observations = face_observations[-obs_to_average:]
        [x1, y1, x2, y2] = [int(x) for x in np.average(face_observations, axis=0)]

        # Crop it.
        frame = frame[y1:y2, x1:x2]

    return frame


def frame_from_screen_cap(x, y, width, height):

    # # Screen capture.
    with mss() as sct:
        monitor = {
            "top": int(x),
            "left": int(y),
            "width": int(width),
            "height": int(height),
        }
        frame = np.array(sct.grab(monitor))
        # Remove the alpha channel.
        frame = frame[:, :, :3]
        return frame


def calculate_fps():
    global frame_counter, last_time

    try:
        frame_counter

    except NameError:
        frame_counter = 0
        last_time = time.time()

    frame_counter += 1
    if frame_counter % 10 == 0:

        print("fps: ", frame_counter / (time.time() - last_time))
        frame_counter = 0
        last_time = time.time()


def dump_frame_to_obs_virtual_cam(frame):
    global virtual_cam

    height, width, layers = frame.shape

    # Make the webcam be bigger.
    width = width * 2
    height = height * 2

    try:
        virtual_cam
    except NameError:

        print("Make sure you're running OBS!")

        virtual_cam = pyvirtualcam.Camera(width=width, height=height, fps=30)
        print(f"Using virtual camera: {virtual_cam.device}")

    updated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    updated_frame = cv2.resize(updated_frame, (height, width))

    virtual_cam.send(updated_frame)
    virtual_cam.sleep_until_next_frame()


def cleanup(signum=None, frame=None):
    """
    Cleanup when we send C-c
    """

    try:
        virtual_cam.close()
    except:
        pass
    vid.release()
    cv2.destroyAllWindows()
    print("Cleanup complete")


signal.signal(signal.SIGINT, cleanup)

while True:

    calculate_fps()

    # Capture the video frame by frame
    # ret, frame = vid.read()

    frame = frame_from_screen_cap(0, 0, 1024, 1400)

    # background removal with rembg
    # _is_success, buffer = cv2.imencode(".png", frame)
    # result = bg.remove(buffer)
    # frame = Image.open(io.BytesIO(result)).convert("RGBA")
    # frame = np.array(frame)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # Background removal with detectron2
    # people_masks = detection.predict(frame)
    # for mask in people_masks:
    #     mask = mask.numpy()
    #     # Mask out everything that's false.
    #     frame[~mask, :] = [0, 0, 0]

    frame = crop_face(frame)

    frame = anime_gan.anime(frame)

    dump_frame_to_obs_virtual_cam(frame)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cleanup()
