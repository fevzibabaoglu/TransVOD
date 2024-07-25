# Modified by Lu He
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_single as T
from torch.utils.data.dataset import ConcatDataset

import os
import numpy as np
from PIL import Image, ImageDraw
from imgaug import augmenters as iaa
import imgaug.augmentables.bbs as ia_bbs

class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        # idx若为675834，则img_id为675835(img_id=idx+1)
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = self.get_image(path)

        image_id = img_id
        target = {'image_id': image_id, 'annotations': target}

        img, target = self.prepare(img, target)

        # custom transformation
        img, target = custom_transform(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # save the images to a folder for debugging
        # save_image_with_bboxes(img, target, rf"C:\Users\STAJYER\Desktop\transvod\data_augmentation\TransVOD\a\images\{idx}.jpg")

        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train_vid' or image_set == "train_det" or image_set == "train_joint":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([600], max_size=1000),
            normalize,
        ])
    
    if image_set == 'custom_dataset':
        return T.Compose([
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.vid_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        # "train_joint": [(root / "Data" / "DET", root / "annotations" / 'imagenet_det_30plus1cls_vid_train.json'), (root / "Data" / "VID", root / "annotations_10true" / 'imagenet_vid_train.json')],
        # "train_det": [(root / "Data" / "DET", root / "annotations" / 'imagenet_det_30plus1cls_vid_train.json')],
        # "train_vid": [(root / "Data" / "VID", root / "annotations" / 'imagenet_vid_train.json')],
        # "train_joint": [(root / "Data" , root / "annotations" / 'imagenet_vid_train_joint_30.json')],
        # "val": [(root / "Data" / "VID", root / "annotations" / 'imagenet_vid_val.json')],
        "custom_dataset": [(root / "Data" / "train", root / "annotations" / '_annotations.coco.json')],
    }
    datasets = []
    for (img_folder, ann_file) in PATHS[image_set]:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks, cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

    

def custom_transform(image, bbs):
    seq = iaa.Sequential([
        iaa.AddToHueAndSaturation((-10, 10)),
        iaa.Fliplr(0.5),
        iaa.Affine(
            rotate=(-20, 20),
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
        ),
        iaa.SomeOf((0, 1), [
            iaa.Cutout(size=0.2),
            iaa.Dropout(p=(0, 0.2)),
            iaa.SaltAndPepper(0.1, per_channel=True),
        ]),
        iaa.OneOf([
            iaa.Clouds(),
            iaa.Fog(),
            iaa.Rain(drop_size=(0.10, 0.20)),
            iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.007, 0.03)),
            iaa.imgcorruptlike.Spatter(severity=2),
        ]),
        iaa.Sometimes(
            0.5,
            iaa.Sequential([
                iaa.MotionBlur(k=5, angle=[-30, 30]),
                iaa.AdditiveGaussianNoise(scale=(10, 40)),
            ], random_order=True),
            iaa.pillike.FilterDetail(),
        )
    ], random_order=True)

    image_np = np.array(image)

    bbs_imgaug = []
    for box in bbs['boxes']:
        x1, y1, x2, y2 = box
        bbs_imgaug.append(ia_bbs.BoundingBox(x1=x1.item(), y1=y1.item(), x2=x2.item(), y2=y2.item()))
    bbs_on_image = ia_bbs.BoundingBoxesOnImage(bbs_imgaug, shape=image_np.shape)

    image_aug, bbs_aug = seq(image=image_np, bounding_boxes=bbs_on_image)

    image_aug_pil = Image.fromarray(image_aug)

    augmented_boxes = []
    for bb in bbs_aug.bounding_boxes:
        augmented_boxes.append([bb.x1, bb.y1, bb.x2, bb.y2])
    augmented_boxes_tensor = torch.tensor(augmented_boxes, dtype=torch.float32)
    bbs['boxes'] = augmented_boxes_tensor

    return image_aug_pil, bbs

def save_image_with_bboxes(image_tensor, bboxes_tensor, output_path):
    # Convert the image tensor to numpy array
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # Change channel order from (C, H, W) to (H, W, C)
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)  # Scale pixel values from [0, 1] to [0, 255] and convert to uint8

    # Convert numpy array to PIL image
    image_pil = Image.fromarray(image_np)

    # Create a drawing context
    draw = ImageDraw.Draw(image_pil)

    # Plot each bounding box
    for bbox in bboxes_tensor['boxes']:
        x1, y1, x2, y2 = bbox.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    # Save the image with bounding boxes
    image_pil.save(output_path)
