from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import random
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math

class CTDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    if self.opt.mosaic and random.randint(0,1):
      img, labels, bboxes = self.load_mosaic(index)
    else:
      img, labels, bboxes = self.load_image(index)

    img_id = self.images[index]
    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w

    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      # # 先扩充小目标
      # if self.opt.small:
      #   small_object_list = list()
      #   for k in range(num_objs):
      #     ann = anns[k]
      #     bbox = self._coco_box_to_bbox(ann['bbox'])
      #     ann_h, ann_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      #     if self.issmallobject(ann_h, ann_w):
      #       small_object_list.append(k)
      #
      #   for k in small_object_list:
      #     ann = anns[k]
      #     new_ann = self.create_copy_ann(img.shape[0], img.shape[1], ann, anns)
      #     if new_ann != None:
      #       img = self.add_patch_in_img(new_ann, ann, img)
      #       anns.append(new_ann)
      #       num_objs = num_objs + 1

      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1

    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input,
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    for k, (bbox, cls_id) in enumerate(zip(bboxes, labels)):
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret

  def issmallobject(self, h, w):
    if h * w <= 64 * 64:
      return True
    else:
      return False

  def create_copy_ann(self, h, w, ann, anns):
    bbox = self._coco_box_to_bbox(ann['bbox']).astype(np.int)
    ann_h, ann_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    #TODO 尝试次数设置的20，可能会增加训练时间
    for i in range(20):
      random_x, random_y = np.random.randint(int(ann_w / 2), int(w - ann_w / 2)), \
                           np.random.randint(int(ann_h / 2), int(h - ann_h / 2))
      xmin, ymin = random_x - ann_w / 2, random_y - ann_h / 2
      xmax, ymax = xmin + ann_w, ymin + ann_h
      if xmin < 0 or xmax > w or ymin < 0 or ymax > h:
        continue
      new_ann = ann
      new_ann['bbox'] = np.array([xmin, ymin, xmax - xmin, ymax - ymin], dtype=np.int)

      if self.overlap(new_ann, anns):
        return new_ann
    return None

  def overlap(self, new_ann, anns):
    if new_ann is None:
      return False
    bboxA = self._coco_box_to_bbox(new_ann['bbox']).astype(np.int)
    for ann in anns:
      bboxB = self._coco_box_to_bbox(ann['bbox']).astype(np.int)
      left_max = max(bboxA[0], bboxB[0])
      top_max = max(bboxA[1], bboxB[1])
      right_min = min(bboxA[2], bboxB[2])
      bottom_min = min(bboxA[3], bboxB[3])
      inter = max(0, (right_min - left_max)) * max(0, (bottom_min - top_max))
      if inter != 0:
        return True
    return False

  def add_patch_in_img(self, new_ann, ann, img):
    bbox = self._coco_box_to_bbox(ann['bbox']).astype(np.int)
    bboxNew = self._coco_box_to_bbox(new_ann['bbox']).astype(np.int)
    img[bboxNew[1]:bboxNew[3], bboxNew[0]:bboxNew[2], :] = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    return img

  def load_image(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    labels = np.array([self.cat_ids[anno['category_id']] for anno in anns])
    bboxes = np.array([anno['bbox'] for anno in anns], dtype=np.float32)
    if len(bboxes) == 0:
      bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
      labels = np.array([[0]])
    bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy
    img = cv2.imread(img_path)
    return img, labels, bboxes

  def load_mosaic(self, index):
    s = 480
    labels_result = []
    bboxes_result = []
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    indices = [index] + [random.randint(0, len(self.images) - 1) for _ in range(3)]  # 3 additional image indices
    #遍历进行拼接
    for i, index in enumerate(indices):
      img, labels, bboxes = self.load_image(index)
      h, w = img.shape[0], img.shape[1]
      if i == 0:  # top left
        # 创建马赛克图像
        img_result = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
        # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
        x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
        # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)
        x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
      elif i == 1:  # top right
        # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
        x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
        # 计算截取的图像区域信息(以xc,yc为第二张图像的左下角坐标填充到马赛克图像中，丢弃越界的区域)
        x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
      elif i == 2:  # bottom left
        # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
        x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
        # 计算截取的图像区域信息(以xc,yc为第三张图像的右上角坐标填充到马赛克图像中，丢弃越界的区域)
        x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
      elif i == 3:  # bottom right
        # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
        x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
        # 计算截取的图像区域信息(以xc,yc为第四张图像的左上角坐标填充到马赛克图像中，丢弃越界的区域)
        x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

      #填充到mosaic中去
      img_result[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
      # 计算pad(图像边界与马赛克边界的距离，越界的情况为负值)
      padw = x1a - x1b
      padh = y1a - y1b

      # Labels 获取对应拼接图像的labels信息
      # [class_index, x_center, y_center, w, h]

      bboxes_temp = bboxes.copy()  # 深拷贝，防止修改原数据
      if bboxes.size > 0:  # Normalized xywh to pixel xyxy format
        # 计算标注数据在马赛克图像中的坐标(绝对坐标)
        bboxes_temp[:, 0] = w * (bboxes[:, 0] - bboxes[:, 2] / 2) + padw  # xmin
        bboxes_temp[:, 1] = h * (bboxes[:, 1] - bboxes[:, 3] / 2) + padh  # ymin
        bboxes_temp[:, 2] = w * (bboxes[:, 0] + bboxes[:, 2] / 2) + padw  # xmax
        bboxes_temp[:, 3] = h * (bboxes[:, 1] + bboxes[:, 3] / 2) + padh  # ymax
      bboxes_result.append(bboxes_temp)
      labels_temp = labels.copy()
      #添加类别信息
      labels_result.append(labels_temp)

    if len(labels_result):
        labels_result = np.concatenate(labels_result, 0)

    if len(bboxes_result):
      bboxes_result = np.concatenate(bboxes_result, 0)
      np.clip(bboxes_result[:, 1:], 0, 2 * s, out=labels_result[:, 1:])

    return img_result, labels_result, bboxes_result