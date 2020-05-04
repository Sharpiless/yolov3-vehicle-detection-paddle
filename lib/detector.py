
import ppdet.utils.checkpoint as checkpoint
from ppdet.utils.cli import ArgsParser
from ppdet.utils.eval_utils import parse_fetches
from ppdet.core.workspace import load_config, create
from paddle import fluid
import os
import cv2
import glob

from ppdet.utils.coco_eval import bbox2out, mask2out, get_category_info
from lib.classifier import CarClassifier

import numpy as np
from PIL import Image
from PIL import ImageFont, ImageDraw

font_path = r'./simsun.ttc'
font = ImageFont.truetype(font_path, 16)


def putText(img, text, x, y, color=(0, 0, 255)):

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    b, g, r = color
    a = 0
    draw.text((x, y), text, font=font, fill=(b, g, r, a))
    img = np.array(img_pil)
    return img


class VehicleDetector(object):

    def __init__(self):

        self.size = 608

        self.draw_threshold = 0.1

        self.cfg = load_config('./configs/vehicle_yolov3_darknet.yml')

        self.place = fluid.CUDAPlace(
            0) if self.cfg.use_gpu else fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)

        self.model = create(self.cfg.architecture)

        self.classifier = CarClassifier()

        self.init_params()

    def draw_bbox(self, image, catid2name, bboxes, threshold):

        raw = image.copy()

        for dt in np.array(bboxes):

            catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
            if score < threshold or catid == 6:
            # if score < threshold:
                continue

            xmin, ymin, w, h = bbox
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmin + w)
            ymax = int(ymin + h)
            roi = raw[ymin:ymax, xmin:xmax].copy()
            label = self.classifier.predict(roi)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
            image = putText(image, label, 0, 10, color=(255, 50, 0))
            print(label)
            print()

        return image

    def init_params(self):

        startup_prog = fluid.Program()
        infer_prog = fluid.Program()
        with fluid.program_guard(infer_prog, startup_prog):
            with fluid.unique_name.guard():
                inputs_def = self.cfg['TestReader']['inputs_def']
                inputs_def['iterable'] = True
                feed_vars, loader = self.model.build_inputs(**inputs_def)
                test_fetches = self.model.test(feed_vars)
        infer_prog = infer_prog.clone(True)

        self.exe.run(startup_prog)
        if self.cfg.weights:
            checkpoint.load_params(self.exe, infer_prog, self.cfg.weights)

        extra_keys = ['im_info', 'im_id', 'im_shape']
        self.keys, self.values, _ = parse_fetches(
            test_fetches, infer_prog, extra_keys)
        dataset = self.cfg.TestReader['dataset']
        anno_file = dataset.get_anno()
        with_background = dataset.with_background
        use_default_label = dataset.use_default_label

        self.clsid2catid, self.catid2name = get_category_info(anno_file, with_background,
                                                              use_default_label)

        is_bbox_normalized = False
        if hasattr(self.model, 'is_bbox_normalized') and \
                callable(self.model.is_bbox_normalized):
            is_bbox_normalized = self.model.is_bbox_normalized()

        self.is_bbox_normalized = is_bbox_normalized

        self.infer_prog = infer_prog

    def process_img(self, img):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        shape = img.shape[:2]

        img = cv2.resize(img, (self.size, self.size))

        # RBG img [224,224,3]->[3,224,224]
        img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
        img_mean = np.array(mean).reshape((3, 1, 1))
        img_std = np.array(std).reshape((3, 1, 1))
        img -= img_mean
        img /= img_std

        img = img.astype('float32')
        img = np.expand_dims(img, axis=0)

        shape = np.expand_dims(np.array(shape), axis=0)
        im_id = np.zeros((1, 1), dtype=np.int64)

        return img, im_id, shape

    def detect(self, img):

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw = img.copy()
        img, im_id, shape = self.process_img(img=img)
        outs = self.exe.run(self.infer_prog,
                            feed={'image': img, 'im_size': shape, 'im_id': im_id},
                            fetch_list=self.values,
                            return_numpy=False)
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(self.keys, outs)
        }

        bbox_results = bbox2out(
            [res], self.clsid2catid, self.is_bbox_normalized)

        result = self.draw_bbox(raw, self.catid2name,
                                bbox_results, self.draw_threshold)

        return result
