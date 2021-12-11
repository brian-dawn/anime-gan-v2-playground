# Stuff for face tracking/cropping.
from facenet_pytorch import InceptionResnetV1, MTCNN

# face tracking
mtcnn = MTCNN(keep_all=True, device="cuda")


def detect_faces(frame):

    # Detect faces
    boxes, _ = mtcnn.detect(frame)
    return boxes
