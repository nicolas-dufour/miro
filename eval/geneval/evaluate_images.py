"""
Evaluate generated images using Mask2Former (or other object detector model)
and CLIP for color classification.

Adapted from GenEval (https://github.com/djghosh13/geneval).
Compatible with mmdet 3.x + mmcv 2.x + mmengine.
"""

import argparse
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

import mmdet
from mmdet.apis import inference_detector, init_detector

import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc

zsc.tqdm = lambda it, *args, **kwargs: it

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLORS = [
    "red", "orange", "yellow", "green", "blue",
    "purple", "pink", "brown", "black", "white",
]


def _find_mmdet_config(config_name):
    """Find mmdet config file, supporting both mmdet 2.x and 3.x layouts."""
    # mmdet 3.x renamed configs, e.g.:
    #   2.x: mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco
    #   3.x: mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco
    MMDET3_CONFIG_RENAMES = {
        "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco":
            "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco",
    }
    names_to_try = [config_name]
    if config_name in MMDET3_CONFIG_RENAMES:
        names_to_try.append(MMDET3_CONFIG_RENAMES[config_name])

    mmdet_dir = os.path.dirname(mmdet.__file__)
    candidates = []
    for name in names_to_try:
        candidates.extend([
            os.path.join(mmdet_dir, ".mim", "configs", "mask2former", f"{name}.py"),
            os.path.join(mmdet_dir, "..", "configs", "mask2former", f"{name}.py"),
            os.path.join(mmdet_dir, "configs", "mask2former", f"{name}.py"),
        ])
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Could not find mmdet config '{config_name}.py'. "
        f"Searched: {candidates}. "
        f"You may need to pass --model-config explicitly."
    )


def timed(fn):
    def wrapper(*args, **kwargs):
        startt = time.time()
        result = fn(*args, **kwargs)
        endt = time.time()
        print(
            f"Function {fn.__name__!r} executed in {endt - startt:.3f}s",
            file=sys.stderr,
        )
        return result
    return wrapper


def _parse_detection_result(result, num_classes):
    """
    Parse inference_detector output into per-class bbox and segm lists.
    Supports both mmdet 2.x (tuple of lists) and 3.x (DetDataSample).
    Returns: (bbox_per_class, segm_per_class) where each is a list indexed by class.
    """
    # mmdet 2.x returns a tuple: (bbox_list, segm_list)
    if isinstance(result, tuple):
        bbox = result[0]
        segm = result[1] if len(result) > 1 else None
        return bbox, segm

    # mmdet 3.x returns a DetDataSample
    pred = result.pred_instances
    bboxes = pred.bboxes.cpu().numpy()     # (N, 4)
    scores = pred.scores.cpu().numpy()     # (N,)
    labels = pred.labels.cpu().numpy()     # (N,)
    masks = pred.masks.cpu().numpy() if hasattr(pred, "masks") and pred.masks is not None else None  # (N, H, W)

    # Convert to per-class format: bbox_per_class[cls] = (M, 5) with score appended
    bbox_per_class = []
    segm_per_class = []
    for cls_idx in range(num_classes):
        cls_mask = labels == cls_idx
        if cls_mask.any():
            cls_bboxes = bboxes[cls_mask]
            cls_scores = scores[cls_mask]
            bbox_per_class.append(
                np.hstack([cls_bboxes, cls_scores[:, None]])
            )
            if masks is not None:
                segm_per_class.append([m for m, keep in zip(masks[cls_mask], [True] * cls_mask.sum()) if keep])
            else:
                segm_per_class.append(None)
        else:
            bbox_per_class.append(np.zeros((0, 5), dtype=np.float32))
            segm_per_class.append(None)

    return bbox_per_class, segm_per_class


class GenevalEvaluator:
    """Encapsulates all models and parameters for GenEval evaluation."""

    def __init__(self, model_config, model_path, options=None):
        self.options = options or {}
        self.threshold = float(self.options.get("threshold", 0.3))
        self.counting_threshold = float(self.options.get("counting_threshold", 0.9))
        self.max_objects = int(self.options.get("max_objects", 16))
        self.nms_threshold = float(self.options.get("max_overlap", 1.0))
        self.position_threshold = float(self.options.get("position_threshold", 0.1))

        self.object_detector, self.clip_model, self.transform, self.tokenizer = (
            self._load_models(model_config, model_path)
        )

        classnames_path = os.path.join(os.path.dirname(__file__), "object_names.txt")
        with open(classnames_path) as f:
            self.classnames = [line.strip() for line in f]

        self.color_classifiers = {}

    @timed
    def _load_models(self, model_config, model_path):
        object_detector_name = self.options.get(
            "model", "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"
        )
        ckpt_path = os.path.join(model_path, f"{object_detector_name}.pth")
        object_detector = init_detector(model_config, ckpt_path, device=DEVICE)

        clip_arch = self.options.get("clip_model", "ViT-L-14")
        clip_model, _, transform = open_clip.create_model_and_transforms(
            clip_arch, pretrained="openai", device=DEVICE
        )
        tokenizer = open_clip.get_tokenizer(clip_arch)

        return object_detector, clip_model, transform, tokenizer

    def _color_classification(self, image, bboxes, classname):
        if classname not in self.color_classifiers:
            self.color_classifiers[classname] = zsc.zero_shot_classifier(
                self.clip_model,
                self.tokenizer,
                COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object",
                ],
                DEVICE,
            )
        clf = self.color_classifiers[classname]
        dataloader = torch.utils.data.DataLoader(
            ImageCrops(image, bboxes, self.transform, self.options),
            batch_size=16,
            num_workers=4,
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(self.clip_model, clf, dataloader, DEVICE)
            return [COLORS[index.item()] for index in pred.argmax(1)]

    def _relative_position(self, obj_a, obj_b):
        """Give position of A relative to B, factoring in object dimensions."""
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        revised_offset = np.maximum(
            np.abs(offset) - self.position_threshold * (dim_a + dim_b), 0
        ) * np.sign(offset)
        if np.all(np.abs(revised_offset) < 1e-3):
            return set()
        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5:
            relations.add("left of")
        if dx > 0.5:
            relations.add("right of")
        if dy < -0.5:
            relations.add("above")
        if dy > 0.5:
            relations.add("below")
        return relations

    def _evaluate(self, image, objects, metadata):
        """
        Evaluate given image using detected objects on the metadata specifications.
        Include clauses are combined with AND, exclude clauses with OR.
        """
        correct = True
        reason = []
        matched_groups = []
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])[:req["count"]]
            if len(found_objects) < req["count"]:
                correct = matched = False
                reason.append(
                    f"expected {classname}>={req['count']}, found {len(found_objects)}"
                )
            else:
                if "color" in req:
                    colors = self._color_classification(image, found_objects, classname)
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found "
                            + f"{colors.count(req['color'])} {req['color']}; and "
                            + ", ".join(
                                f"{colors.count(c)} {c}" for c in COLORS if c in colors
                            )
                        )
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = self._relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found "
                                        + f"{' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        for req in metadata.get("exclude", []):
            classname = req["class"]
            if len(objects.get(classname, [])) >= req["count"]:
                correct = False
                reason.append(
                    f"expected {classname}<{req['count']}, found {len(objects[classname])}"
                )
        return correct, "\n".join(reason)

    def evaluate_image(self, filepath, metadata):
        result = inference_detector(self.object_detector, filepath)
        bbox, segm = _parse_detection_result(result, len(self.classnames))
        image = ImageOps.exif_transpose(Image.open(filepath))
        detected = {}
        confidence_threshold = (
            self.threshold if metadata["tag"] != "counting" else self.counting_threshold
        )
        for index, classname in enumerate(self.classnames):
            ordering = np.argsort(bbox[index][:, 4])[::-1]
            ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]
            ordering = ordering[:self.max_objects].tolist()
            detected[classname] = []
            while ordering:
                max_obj = ordering.pop(0)
                segm_for_class = segm[index] if segm is not None else None
                detected[classname].append(
                    (bbox[index][max_obj], None if segm_for_class is None else segm_for_class[max_obj])
                )
                ordering = [
                    obj
                    for obj in ordering
                    if self.nms_threshold == 1
                    or compute_iou(bbox[index][max_obj], bbox[index][obj]) < self.nms_threshold
                ]
            if not detected[classname]:
                del detected[classname]
        is_correct, reason = self._evaluate(image, detected, metadata)
        return {
            "filename": filepath,
            "tag": metadata["tag"],
            "prompt": metadata["prompt"],
            "correct": is_correct,
            "reason": reason,
            "metadata": json.dumps(metadata),
            "details": json.dumps(
                {key: [box.tolist() for box, _ in value] for key, value in detected.items()}
            ),
        }

    def evaluate_directory(self, imagedir, outfile):
        full_results = []
        for subfolder in os.listdir(imagedir):
            folderpath = os.path.join(imagedir, subfolder)
            if not os.path.isdir(folderpath) or not subfolder.isdigit():
                continue
            with open(os.path.join(folderpath, "metadata.jsonl")) as fp:
                metadata = json.load(fp)
            for imagename in os.listdir(os.path.join(folderpath, "samples")):
                imagepath = os.path.join(folderpath, "samples", imagename)
                if not os.path.isfile(imagepath) or not re.match(r"\d+\.png", imagename):
                    continue
                result = self.evaluate_image(imagepath, metadata)
                full_results.append(result)
        if os.path.dirname(outfile):
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, "w") as fp:
            pd.DataFrame(full_results).to_json(fp, orient="records", lines=True)
        return full_results


class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects, transform, options):
        self._image = image.convert("RGB")
        bgcolor = options.get("bgcolor", "#999")
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects
        self._transform = transform
        self._crop = options.get("crop", "1") == "1"

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (
                index, self._image.size[::-1], mask.shape,
            )
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        if self._crop:
            image = image.crop(box[:4])
        return (self._transform(image), 0)


def compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn([
        max(box_a[0], box_b[0]),
        max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]),
        min(box_a[3], box_b[3]),
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def parse_args():
    parser = argparse.ArgumentParser(description="GenEval: Evaluate generated images")
    parser.add_argument("imagedir", type=str, help="Directory containing generated images")
    parser.add_argument("--outfile", type=str, default="results.jsonl")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="./")
    parser.add_argument("--options", nargs="*", type=str, default=[])
    args = parser.parse_args()
    args.options = dict(opt.split("=", 1) for opt in args.options)
    if args.model_config is None:
        config_name = args.options.get(
            "model", "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"
        )
        args.model_config = _find_mmdet_config(config_name)
    return args


def main():
    args = parse_args()
    assert DEVICE == "cuda", "GenEval evaluation requires CUDA"
    evaluator = GenevalEvaluator(args.model_config, args.model_path, args.options)
    evaluator.evaluate_directory(args.imagedir, args.outfile)


if __name__ == "__main__":
    main()
