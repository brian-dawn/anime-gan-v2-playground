# Stuff for masking

import matplotlib.pyplot as plt

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

# Detectron2 crap.
model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"


model = model_zoo.get(model_name, trained=True)
cfg = get_cfg()
cfg.device = "cuda"
cfg.merge_from_file(model_zoo.get_config_file(model_name))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
predictor = DefaultPredictor(cfg)

metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
class_catalog = metadata.thing_classes


def predict(frame):

    outputs = predictor(frame[..., ::-1])

    instances = outputs["instances"]
    from pprint import pprint

    filtered_instances = []

    for index, ((y1, x1, y2, x2), score, mask, klass) in enumerate(
        zip(
            instances.pred_boxes.to("cpu"),
            instances.scores.to("cpu"),
            instances.pred_masks.to("cpu"),
            instances.pred_classes,
        )
    ):
        class_name = class_catalog[klass.item()]

        if class_name == "person":
            filtered_instances.append(mask)

    # v = Visualizer(
    #     frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
    # )
    # print(len(filtered_instances))

    return filtered_instances

    # out = v.draw_instance_predictions(instances.to("cpu"))

    # return out.get_image()[..., ::-1][..., ::-1]


if __name__ == "__main__":
    import cv2
    import numpy as np

    vid = cv2.VideoCapture(1)

    while True:

        ret, im = vid.read()

        outputs = predictor(im[..., ::-1])
        v = Visualizer(
            im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(20, 10))
        cv2.imshow("frame", out.get_image()[..., ::-1][..., ::-1])

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # predict(frame)
