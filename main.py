import os
import numpy as np
import types
import time
import torch
import torchvision
import xml.etree.ElementTree as ET  # xml 다루기, xml은 트리구조로 되어있음
from PIL import Image
from math import sqrt
from itertools import product  # 디폴트박스 출력 클래스에서 쓰임
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from numpy import random
import matplotlib.cm as cm
import torchvision.models as models
from torchvision.models.detection import ssd300_vgg16
from torch.autograd import Function  # detect 클래스 상속
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from collections import Counter

#####################################################################################################
def make_datapath_list(rootpath):
    """
    파일 경로를 저장한 이미지 리스트, 어노테이션 리스트 작성
    ----------
    <Parameters>
    rootpath : str => 데이터 폴더의 경로
    -------
    <Returns>
    train_img_list, train_anno_list, val_img_list, val_anno_list => 데이터 경로 저장 리스트
    img_list = ['~.jpg', '~.jpg', ... '~.jpg', '~.jpg',] (string)
    anno_list = ['~.txt', '~.txt', ... '~.txt', '~.txt',] (string)
    """

    jpg_name = '우산'
    jpg_total_num = 6300

    train_path = os.path.join(rootpath + '우산_test/')                    # 훈련 및 검증 파일명 리스트 텍스트 파일
    val_path = os.path.join(rootpath + '우산_val/')                       # jpeg 이미지의 파일명들 작성되어 있음

    train_img_list = list()                                              # 훈련 데이터의 이미지 파일과 어노테이션 파일을 저장할 경로 리스트 생성
    train_anno_list = list()

    for i in range(jpg_total_num):
        img_path = os.path.join(train_path, f'{jpg_name}_{i}.jpg')
        anno_path = os.path.join(train_path, f'{jpg_name}_{i}.txt')

        if os.path.exists(img_path) and os.path.exists(anno_path):
            train_img_list.append(img_path)
            train_anno_list.append(anno_path)

    val_img_list = list()
    val_anno_list = list()

    for i in range(jpg_total_num):
        img_path = os.path.join(rootpath, f'{jpg_name}_{i}.jpg')
        anno_path = os.path.join(rootpath, f'{jpg_name}_{i}.txt')

        if os.path.exists(img_path) and os.path.exists(anno_path):
            val_img_list.append(img_path)
            val_anno_list.append(anno_path)
###################
    imgpath_template = os.path.join(rootpath, 'data/VOCdevkit/VOC2012/JPEGImages/')                # 이미지 파일과 어노테이션 파일의 경로 템플릿 작성
    annopath_template = os.path.join(rootpath, 'data/VOCdevkit/VOC2012/Annotations/')

    train_id_names = os.path.join(rootpath + 'data/VOCdevkit/VOC2012/ImageSets/Main/train.txt')    # 훈련 및 검증 파일명 리스트 텍스트 파일
    val_id_names = os.path.join(rootpath + 'data/VOCdevkit/VOC2012/ImageSets/Main/val.txt')        # jpeg 이미지의 파일명들 작성되어 있음


    # for line in open(train_id_names):                                       # 이미지들의 파일명 조회
    #     file_id = line.strip()                                              # 공백 및 줄바꿈 제거
    #     img_path = imgpath_template + file_id + '.jpg'                      # 이미지 경로
    #     anno_path = annopath_template + file_id + '.xml'                    # 어노테이션 경로
    #     train_img_list.append(img_path)                                     # 리스트에 추가
    #     train_anno_list.append(anno_path)                                   # 리스트에 추가
    #
    # for line in open(val_id_names):
    #     file_id = line.strip()                                              # 공백과 줄바꿈 제거
    #     img_path = imgpath_template + file_id + '.jpg'                      # 이미지 경로
    #     anno_path = annopath_template + file_id + '.xml'                    # 어노테이션 경로
    #     val_img_list.append(img_path)                                       # 리스트에 추가
    #     val_anno_list.append(anno_path)                                     # 리스트에 추가

    return train_img_list, train_anno_list, val_img_list, val_anno_list

#####################################################################################################
class Anno_xml2list():
    """
    txt 형식의 어노테이션을 리스트 형식으로 변환하는 클래스
    ----------
    <Attributes>
    classes : 리스트, VOC의 클래스명을 저장한 리스트
    """

    def __init__(self, classes):  # 클래스명 리스트 저장
        self.classes = classes

    def __call__(self, txt_path, width, height):
        """
        XML 형식의 어노테이션을 리스트 형식으로 변환
        ----------
        <Parameters>
        txt_path : str => xml 파일 경로
        width : int => 이미지 넓이
        height : int => 이미지 높이
        -------
        <Returns>
        np.array, [[xmin, ymin, xmax, ymax, label_ind], ... ]
        => 한 이미지의 어노테이션 데이터를 저장한 리스트. 한 이미지에 존재하는 물체 수만큼의 요소를 가짐.
        """

        img_anno = []                                                   # 한 이미지 내 모든 물체의 어노테이션을 이 리스트에 저장,

        if 'txt' in txt_path:
            with open(txt_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    cx, cy, w, h = map(float, parts[1:])
                    xmin = cx - (w/2)
                    xmax = cx + (w/2)
                    ymin = cy - (h/2)
                    ymax = cy + (h / 2)
                    label = int(parts[0])+20
                    img_anno.append([xmin, ymin, xmax, ymax, label])
            return np.array(img_anno)                                   # [[xmin, ymin, xmax, ymax, label_ind], ... ]
        else:
            xml = ET.parse(txt_path).getroot()                          # xml파일을 파싱해서 ElementTree 객체를 생성하고 루트 요소 반환
            for obj in xml.iter('object'):                              # 이미지 내 물체의 수 만큼 반복
                difficult = int(obj.find('difficult').text)             # annotation에서 difficult가 1로 설정된 것은 제외
                if difficult == 1:
                    continue

                obj_anno = []                                           # 한 물체의 어노테이션을 저장하는 리스트
                name = obj.find('name').text.lower().strip()            # 물체 이름
                bbox = obj.find('bndbox')                               # 바운딩 박스 정보
                b_pts = ['xmin', 'ymin', 'xmax', 'ymax']                # for문 활용해 찾을 어노테이션 정보 저장 리스트

                for b_pt in (b_pts):                                    # 어노테이션의 xmin, ymin, xmax, ymax를 취득하고, 0 ~ 1로 규격화,['xmin', 'ymin', 'xmax', 'ymax']
                    cur_pixel = int(bbox.find(b_pt).text) - 1           # VOC는 원점이 (1, 1)이므로 1을 빼서 (0, 0)으로 한다.

                    if b_pt == 'xmin' or b_pt == 'xmax':                # 폭, 높이로 규격화
                        cur_pixel /= width                              # x 방향의 경우 폭으로 나눔 => 0~1의 좌표
                    else:                                               # y 방향의 경우 높이로 나눔 => 0~1의 좌표
                        cur_pixel /= height                             # 이미지 넓이가 100이고 xmin이 30이면 0.3으로 정규화

                    obj_anno.append(cur_pixel)                          # xmin, ymin, xmax, ymax 순서로 입력

                label_idx = self.classes.index(name)                    # 어노테이션 클래스명 index를 취득하여 추가
                obj_anno.append(label_idx)
                img_anno += [obj_anno]                                  # [xmin, ymin, xmax, ymax, label_ind]를 더한다.

            return np.array(img_anno)                                   # [[xmin, ymin, xmax, ymax, label_ind], ... ]

#####################################################################################################
# 데이터 전처리에 필요한 함수들

def intersect_numpy(box_a, box_b):                              # intersect_numpy는 a는 여러개 box, b는 1개
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])                # box_a Shape: [num_boxes,4] => [num_boxes, [xmin, ymin, xmax, ymax]]
    min_xy = np.maximum(box_a[:, :2], box_b[:2])                # box_b Shape: [4] => [xmin, ymin, xmax, ymax]
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)   # inter[:, 0] : width, inter[:, 1] : height
    return inter[:, 0] * inter[:, 1]                            # return shape [num_boxes, 1] => 넓이

def jaccard_numpy(box_a, box_b):                                # jaccard_numpy a는 여러개 box, b는 1개
    """ jaccard overlap
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a Shape: [num_boxes,4]
        box_b Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], [1]]
    """
    inter = intersect_numpy(box_a, box_b)                                   # inter [num_boxes, 1]
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))    # (ymax - ymin) * (xmax - xmin) = 높이 * 넓이 , [num_boxes, 1]
    area_b = ((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))                # (ymax - ymin) * (xmax - xmin) = 높이 * 넓이 , [1]
    union = area_a + area_b - inter                                         # 두 박스의 넓이 - 교집합 = 전체 넓이
    return inter / union                                                    # 교집합 / 합집합 = IOU, [num_boxes, 1]

class Compose(object):
    """
    전처리 함수들을 리스트 형태로 저장하고 __call__로 실행
    Args:
        transforms (List[Transform]): transforms의 리스트.
    Example:
        data_transform.Compose([
            transforms.CenterCrop(10),
            transforms.ToTensor(),
        ])
    """
    def __init__(self, transforms):                         # transform 함수들 리스트 저장
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):       # 받은 리스트의 함수들 실행
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class ConvertFromInts(object):                              # 이미지의 픽셀 값을 float32으로 변환
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class SubtractMeans(object):                                # 채널별 평균 값을 빼줌
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels

class ToAbsoluteCoords(object):                             # 바운딩 박스의 좌표 [xmin, ymin, xmax, ymax](0~1)를
    def __call__(self, image, boxes=None, labels=None):     # [xmin, ymin, xmax, ymax](0~300)으로 변환
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        return image, boxes, labels

class ToPercentCoords(object):                              # 바운딩 박스의 좌표 [xmin, ymin, xmax, ymax](0~300)를
    def __call__(self, image, boxes=None, labels=None):     # [xmin, ymin, xmax, ymax](0~1)으로 변환
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        return image, boxes, labels

class Resize(object):                                       # 이미지 크기를 300*300으로 조절
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels

class RandomSaturation(object):                                 # saturation(채도)는 색상 깊이이고 원색에 가까울수록 높다(0~255)
    def __init__(self, lower=0.5, upper=1.5):                   # 반반의 확률로 채도에 0.5~1.5의 값을 곱해줌
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower,self.upper)  # image[:, :, 0] = H, image[:, :, 1] = S, image[:, :, 2] = V
        return image, boxes, labels

class RandomHue(object):                                        # HUE(색상)은 색의 종류이고 0도에서 360도로 표현 opencv에서는 0~179,
    def __init__(self, delta=18.0):                             # 색상에 반반확률로 -delta~delta값을 더하고 0~179로 클리핑
        assert delta >= 0.0 and delta <= 179.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta,self.delta)  # image[:, :, 0] = H, image[:, :, 1] = S, image[:, :, 2] = V
            image[:, :, 0][image[:, :, 0] > 179.0] -= 179.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 179.0
        return image, boxes, labels

class RandomLightingNoise(object):                                      # 반반의 확률로 무작위로 채널의 순서를 바꿈
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
        return image, boxes, labels

class SwapChannels(object):                                 # 입력 받은 셔플리스트에 따라 채널의 위치 변경
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

class ConvertColor(object):                                     # BGR의 이미지를 HSV로, HSV의 이미지를 BGR로 변환
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels

class RandomContrast(object):                                               # 반반 랜덤으로 lower ~ upper 사이의 랜덤한 값을 픽셀 값에 곱해줌,
    def __init__(self, lower=0.5, upper=1.5):                               # 클리핑 안됨 0~255 벗어날수도 있음
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):                     # expects float image
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels

class RandomBrightness(object):                                             # 반반 랜덤으로 -delta ~ delta 값을 더해줌,
    def __init__(self, delta=32):                                           # 클리핑 안됨 0~255 벗어날수도 있음
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels

class ToCV2Image(object):                                                   # 넘파이배열 -> 텐서, [H, W, C] -> [C, H, W]
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels

class ToTensor(object):                                                     # 텐서 -> 넘파이배열, [C, H, W] -> [H, W, C]
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels

class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): Expand()를 거쳐서 사이즈가 300*300 ~ 1200*1200, 채널 순서 무작위
        boxes (Tensor): [xmin, ymin, xmax, ymax] (0 ~ 1200)
        labels (Tensor): [num_object, 1]
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (-1, 0.1, 0.3, 0.5, 0.7, 0.9, None)       # 크롭한 이미지와 바운딩 박스의 최소 IOU
                                                                        # -1은 min_iou 상관없이 랜덤
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode == None:  # None이면 원본 그대로
                return image, boxes, labels

            min_iou = mode

            for _ in range(50):
                current_image = image
                w = random.uniform(0.3 * width, width)                      # H, W (0.3 ~ 1) 무작위
                h = random.uniform(0.3 * height, height)                    # 크롭할 이미지의 높이와 넓이

                if h / w < 0.5 or h / w > 2:                                # aspect ratio 가 0.5이상 2 이하여야 됨
                    continue

                left = random.uniform(width - w)                            # crop된 이미지가 들어갈 수 있는 랜덤 좌표(시작좌표)
                top = random.uniform(height - h)

                rect = np.array(
                    [int(left), int(top), int(left + w), int(top + h)])     # 이미지에서 크롭할 부분[xmin, ymin, xmax, ymax] (0~1200)

                overlap = jaccard_numpy(boxes, rect)                        # crop할 이미지와 바운딩 박스간의 IOU계산 , jaccard는 여러개랑 여러개

                if overlap.min() < min_iou:                                 # 여러 물체 중 모두가 최소IOU 이상이면 크롭 진행
                    continue                                                # min_iou= -1이면 무조건 진행

                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]  # 이미지 크롭

                                                                                    # 바운딩 박스의 중심점
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0                       # centers [num_obj, 2(cx, cy)]

                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])          # 크롭된 이미지가 각 바운딩박스의 중심점을 포함하고 있는지, 포함하면 true
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2

                if not mask.all():                      # mask.all() 모든 물체를 포함해야 크롭 진행
                    continue                            # ????????????#  # 크롭된 이미지가 바운딩박스의 중심점을 포함해야 하는지?
                                                        # min_iou가 0.1이면 포함하지 않을텐데 의미가 있나 -> 아님 크롭이미지가 바운딩 박스보다 작을 수 있음
                                                        # mask.any()로 하나만 중심점을 포함하면 나머지 물체의 어노테이션은 지워야하는지?

                current_boxes = boxes[mask, :].copy()   # 현재 바운딩 박스 복사
                current_labels = labels[mask]           # 모든 물체 포함되게 크롭했기 때문에 레이블 그대로

                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])  # 크롭하고나서의 바운딩 좌표 수정
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels

class Expand(object):                       # BGR 각각의 값의 평균을 받음 [b mean, g mean, r mean], 반반의 확률로 적용
                                            # 이미지의 1~4의 비율로 BGR의 평균으로 채우고 300*300이미지 채워 넣기
    def __init__(self, mean):               # ppt 그림 참조
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)                           # ratio 1~4 랜덤
        left = random.uniform(0, width * ratio - width)             # 0~900 랜덤, 원본이미지 시작좌표
        top = random.uniform(0, height * ratio - height)            # 0~900 랜덤

        expand_image = np.zeros((int(height * ratio), int(width * ratio), depth), dtype=image.dtype)
        expand_image[:, :, :] = self.mean                                                       # 전부 채널 평균값으로 채움
        expand_image[int(top):int(top + height), int(left):int(left + width)] = image           # (top, left) 시작좌표로부터 원본이미지 채워넣음
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))                           # 바운딩 박스 좌표 시작좌표 만큼 더해줌
        boxes[:, 2:] += (int(left), int(top))                           # xmin, xmax += left, ymin, ymax += top

        return image, boxes, labels

class RandomMirror(object):                             # 반반의 확률로 좌우 반전, 바운딩 박스도 좌표 조절
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1, :]                   # 수정 image = image[:, ::-1] => image = image[:, ::-1, :]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]    # boxes[:, 0::2] = xmin, xmax, boxes[:, 2::-2] = xmax, xmin

        return image, boxes, classes

class PhotometricDistort(object):                               # 각종 노이즈 추가하고 BGR이미지 반환, 0~255값 벗어날 수 있음, 채널의 순서도 무작위
    def __init__(self):
        self.pd = [
            RandomContrast(),                                   # 반반 랜덤으로 lower(0.5) ~ upper(1.5) 사이의 랜덤한 값을 픽셀 값에 곱해줌
            ConvertColor(current = 'BGR', transform='HSV'),     # BGR의 이미지를 HSV로 변환
            RandomSaturation(),                                 # 반반의 확률로 채도에 0.5~1.5의 값을 곱해줌
            RandomHue(),                                        # 반반의 확률로 색상에 -delta(-18)~delta(18)값을 더하고 0~179로 클리핑
            ConvertColor(current='HSV', transform='BGR'),       # HSV의 이미지를 BGR로 변환
            RandomContrast()  #
        ]
        self.rand_brightness = RandomBrightness()               # 반반 랜덤으로 -delta(-32) ~ delta(32) 값을 더해줌
        self.rand_light_noise = RandomLightingNoise()           # 반반의 확률로 무작위로 채널의 순서를 바꿈

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):                                   # 50%의 확률로
            distort = Compose(self.pd[:-1])                     # 처음부터 끝까지 함수 실행
        else:
            distort = Compose(self.pd[1:])                      # 두번째부터 끝까지 함수 실행
        im, boxes, labels = distort(im, boxes, labels)

        return self.rand_light_noise(im, boxes, labels)

#####################################################################################################
# 입력 영상의 전처리 클래스
class DataTransform():
    """
    이미지와 어노테이션의 전처리 클래스. 훈련과 추론에서 다르게 작동.
    이미지 크기를 300*300으로 한다.
    학습 시 데이터 확장을 수행.
    ----------
    <Attributes>
    input_size : int => 리사이즈 할 크기 300
    color_mean : (B, G, R) => 각 색상채널 평균 값
    """

    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),          # 이미지의 픽셀 값을 int에서 float으로 변환
                ToAbsoluteCoords(),         # 바운딩 박스의 좌표 [xmin, ymin, xmax, ymax](0~1)를 [xmin, ymin, xmax, ymax](0~300)으로 변환
                PhotometricDistort(),       # 각종 노이즈 추가하고 BGR이미지 반환, 0~255값 벗어날 수 있음, 채널의 순서도 무작위
                Expand(color_mean),         # 이미지의 1~4의 비율 크기이고, 값은 BGR의 평균으로 채운 이미지를 만들고 원본이미지 랜덤위치 삽입
                RandomSampleCrop(),         # 랜덤 크롭
                RandomMirror(),             # 반반의 확률로 좌우 반전, 바운딩 박스도 좌표 조절
                ToPercentCoords(),          # 바운딩 박스의 좌표 [xmin, ymin, xmax, ymax](0~300)를 [xmin, ymin, xmax, ymax](0~1)으로 변환
                Resize(input_size),         # input_size×input_size로 크기 변환
                SubtractMeans(color_mean)   # BGR 채널별 평균 값을 빼줌
            ]),
            'val': Compose([
                ConvertFromInts(),          # 이미지의 픽셀 값을 int에서 float으로 변환
                Resize(input_size),         # input_size×input_size로 크기 변환
                SubtractMeans(color_mean)   # BGR 채널별 평균 값을 빼줌
            ])
        }

    def __call__(self, img, phase, boxes, labels):          # data_transform 된 정보 반환
        """
        ----------
        <Parameters>
        img : <class 'numpy.ndarray'> => 하나의 이미지
        phase : 'train' or 'val' => 모드 설정
        boxes :  <class 'numpy.ndarray'> => (2, 4) , (물체의 수, 꼭짓점 좌표)
        labels : <class 'numpy.ndarray'>  => (2,) , 한 이미지에서 물체의 수 만큼의 레이블
        """
        return self.data_transform[phase](img, boxes, labels)

#####################################################################################################
class VOCDataset(data.Dataset):
    """
    VOC2012의 데이터셋을 작성하는 클래스. 파이토치의 dataset 클래스를 상속
    ----------
    img_list : 리스트
        화상 경로를 저장한 리스트
    anno_list : 리스트
        어노테이션 경로를 저장한 리스트
    phase : 'train' or 'test'
        학습 또는 훈련 설정
    transform : object
        전처리 클래스 인스턴스
    transform_anno : object
        xml 어노테이션을 리스트로 변환하는 인스턴스
    """

    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase                                                      # train 또는 val 지정
        self.transform = transform                                              # 이미지 전처리
        self.transform_anno = transform_anno                                    # 어노테이션 데이터를 xml에서 리스트로 변경

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt, h, w

    def pull_item(self, index):
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)
        height, width, channels = img.shape

        anno_file_path = self.anno_list[index]                                                      # 어노테이션 정보를 리스트에 저장
        anno_list = self.transform_anno(anno_file_path, width, height)
        img, boxes, labels = self.transform(img, self.phase, anno_list[:, :4], anno_list[:, 4])     # 전처리 실시

        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)                 # 색상 채널 순서가 BGR이므로 RGB로 순서 변경
                                                                                            # (높이, 폭, 색상 채널)의 순서를 (색상 채널, 높이, 폭)으로 변경
                                                                                            # 넘파이배열 -> 텐서

        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))                             # BBox와 라벨을 세트로 한 np.array 작성, hstack은 수평으로 쌓기
                                                                                            # gt = [[xmin, ymin, xmax, ymax, label], ...[]]
        return img, gt, height, width                                                       # img= 텐서

#####################################################################################################
def od_collate_fn(batch):
    """
    Dataset에서 꺼내는 어노테이션 데이터의 크기는 화상마다 다르다.
    화상 내의 물체 수가 두개이면 (2, 5)사이즈이지만, 세 개이면 (3, 5) 등으로 바뀐다.
    변화에 대응하는 DataLoader를 만드는 collate_fn을 작성한다.
    collate_fn은 파이토치 리스트로 mini batch를 작성하는 함수이디ㅏ.
    미니 배치 분량 화상이 나열된 리스트 변수 batch에 미니 배치 번호를 지정하는
    차원을 가장 앞에 하나 추가하여 리스트 형태를 변형한다.
    batch = (image, annotation)
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])                                  # sample[0]은 이미지 gt
        targets.append(torch.FloatTensor(sample[1]))            # sample[1]은 어노테이션 gt

    imgs = torch.stack(imgs, dim=0)                             # imgs는 미니배치 크기의 리스트
                                                                # 리스트 요소는 torch.Size([3, 300, 300])
                                                                # 이 리스트를 torch.Size([batch_num, 3, 300, 300])의 텐서로 변환

                                                                # targets은 어노테이션의 정답인 gt 리스트
                                                                # 리스트 크기 = 미니 배치 크기
                                                                # targets 리스트의 요소는 [n, 5]텐서
                                                                # n은 화상마다 다르며 화상 속 물체의 수
                                                                # 5는 [xmin, ymin, xmax, ymax, class_index]
                                                                # targets는 리스트 [batch_num, [n, 5]이건 텐서]
    return imgs, targets

#####################################################################################################
def make_vgg():            # 34층에 걸친 vgg 모듈을 작성
    layers = []
    in_channels = 3

    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'MC', 512, 512, 512, 'M', 512, 512, 512]  # vgg 모듈 레이어

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MC':
            layers += [
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]      # ceil은 계산 결과(float)에서 출력 크기의 소수점을 올려 정수로 하는 모드
        else:                                                               # default는 계산 결과(float)에서 출력 크기의 소수점을 버려 정수로 하는 floor 모드
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]                       # ReLU 인수 inplace는 ReLU에 대한 입력을 메모리 상에 유지할 것인지, 혹은
            in_channels = v                                                 # 입력을 재작성 하여 출력으로 바꾼 후 메모리상에 유지하지 않을 것인지를 나타냄.
                                                                            # inplace=True 입력 시 메모리상에 입력을 유지하지 않고, 입력을 재작성 (메모리 절약됨)

                                                                            # 여기까지하면 [512, 19, 19]
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)                # 출력 형태 같음 [512, 19, 19]
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # dilation=6이면 필터 픽셀 사이에 5씩 추가되어 필터가 13크기가 됨.
                                                                            # 필터 크기는 13이지만 9개의 픽셀만 연산, H = ((H + 2p - kernel) / stride) + 1
                                                                            # 따라서 출력은 [1024, 19, 19]
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)       # 출력은 [1024, 19, 19] , classifier
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)

#####################################################################################################
def make_extras():                  # 8층에 걸친 extras 모듈 작성
    layers = []
    in_channels = 1024              # vgg모듈에서 출력된 extra에 입력되는 화상 채널 수

    cfg = [256, 512, 128, 256, 128, 256, 128, 256]      # extra 모듈의 합성곱 층 채널 수를 설정하는 구성(configuration)

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]                     # 출력 [256, 19, 19], conv8_1
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]     # 출력 [512, 10, 10], conv8_2 , classifier
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]                          # 출력 [128, 10, 10], conv9_1
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]     # 출력 [256, 5, 5],   conv9_2 , classifier
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]                          # 출력 [128, 5, 5],   conv10_1
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]                          # 출력 [256, 3, 3],   conv10_2 , classifier
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]                          # 출력 [128, 3, 3],   conv11_1
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]                          # 출력 [256, 1, 1],   conv11_2 , classifier

    return nn.ModuleList(layers)                            # 활성화 함수의 ReLU는 SSD 모듈의 순전파에서 준비하고,
                                                            # extra에서는 준비하지 않음.
                                                            # classifier는 conv4_3, conv7, conv8_2, conv9_2, conv10_2 conv11_2 총 6개의 피쳐맵
#####################################################################################################
def make_loc_conf(num_classes, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):     # 디폴트 박스의 오프셋을 출력하는 loc_layers와
    loc_layers = []                                                     # 디폴트 박스 각 클래스 신뢰도 confidence를 출력하는 conf_layers 작성
    conf_layers = []
                                                                                                            # VGG의 (source1)의 합성곱 층
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0] * 4, kernel_size=3,padding=1)]               # output channels은 bbox의 종류 수 * 4개의 좌표 = [16, 38, 38]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0] * num_classes, kernel_size=3,padding=1)]    # output channels은 bbox의 종류 수 * class 종류 수 = [84, 38, 38]

                                                                                                            # VGG의 최종층(source2)의 합성곱 층
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * 4, kernel_size=3, padding=1)]             # [24, 19, 19]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * num_classes, kernel_size=3, padding=1)]  # [126, 19, 19]

                                                                                                            # extra(source3)의 합성곱 층
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2] * 4, kernel_size=3, padding=1)]              # [24, 10, 10]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2] * num_classes, kernel_size=3, padding=1)]   # [126, 10, 10]

    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3] * 4, kernel_size=3, padding=1)]              # extra（source4)의 합성곱 층
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3] * num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4] * 4, kernel_size=3, padding=1)]              # extra（source5）의 합성곱 층
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4] * num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5] * 4, kernel_size=3, padding=1)]              # extra（source6）의 합성곱 층
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5] * num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

#####################################################################################################
class L2Norm(nn.Module):                                            # convC4_3의 출력을 scale=20의 L2Norm으로 정규화하는 층
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale                                          # 계수 weight의 초깃값으로 설정할 값
        self.reset_parameters()                                     # 파라미터 초기화
        self.eps = 1e-10

    def reset_parameters(self):
        '''결합 파라미터의 scale 크기 값으로 초기화를 실행'''
        nn.init.constant_(self.weight, self.scale)                  # weight 값이 모두 scale(=20)이 된다.

    def forward(self, x):
        '''38*38의 특징량에 대해 512 채널에 걸쳐 제곱합의 루트를 구했다.
        38*38개의 값을 사용하여 각 특징량을 정규화한 후 계수를 곱하여 계산하는 층'''
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps          # 각 채널의 38*38개 특징량의 채널 방향 제곱합을 계산하고
        x = torch.div(x, norm)                                              # 루트를 구해 나누어 정규화한다.
                                                                            # norm의 텐서 사이즈는 torch.Size([batch_num, 1, 38, 38])
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)   # 계수를 곱한다. 계수는 채널마다 하나로, 512개의 계수를 갖는다.
        out = weights * x                                                           # self.weight의 텐서 사이즈는 torch.Size([512])로,
                                                                                    # torch.Size([batch_num, 512, 38, 38])까지 변형한다.
        return out

#####################################################################################################
class DBox(object):                                 # 디폴트 박스를 출력하는 클래스
    def __init__(self, cfg):
        super(DBox, self).__init__()
        self.image_size = cfg['input_size']         # 이미지 크기는 300
        self.feature_maps = cfg['feature_maps']     # [38, 19, …] 각 source의 특징량 맵 크기
        self.num_priors = len(cfg["feature_maps"])  # source 개수 = 6
        self.aspect_ratios = cfg['aspect_ratios']   # 정사각형의 DBox 화면비(종횡비)
        self.scale = cfg['scale']                   # 각 source에서 작은 1:1 defalutBox의 스케일

    def make_dbox_list(self):                           # [8732, 4]의 디폴트 박스 생성
        mean = []
        for idx, f in enumerate(self.feature_maps):     # 'feature_maps': [38, 19, 10, 5, 3, 1]
            for i, j in product(range(f), repeat=2):    # 데카르트 곱, [i, j] = [0,0], [0,1] ... [38, 38] 즉 모든 좌표 Z-scan order
                cx = (j + 0.5) / self.feature_maps[idx] # 각 피쳐맵을 0 ~ 1로 정규화한 후의 픽셀의 중심 좌표
                cy = (i + 0.5) / self.feature_maps[idx] # (cx, cy) * self.image_size  = 원본에서의 좌표

                s_k = self.scale[idx]                   # 'scale' : [0.2, 0.34, 0.48, 0.62, 0.76, 0.9, 1.04]
                mean += [cx, cy, s_k, s_k]              # 화면비 1의 작은 DBox [cx,cy, width, height]

                s_k_prime = sqrt(s_k * self.scale[idx + 1])     # s_k_prime k+1의 scale이 필요한데 마지막 은 다음 scale이 없어서 스케일이 등간격이라 마지막에 같은 값 더해서 생성해줌
                mean += [cx, cy, s_k_prime, s_k_prime]          # 화면비 1의 큰 DBox [cx,cy, width, height]

                                                                        # 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
                for ar in self.aspect_ratios[idx]:                      # 그 외 화면비의 defBox [cx,cy, width, height]
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]    # W:H 가 2:1, 3:1
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]    # W:H 가 1:2, 1:3

        output = torch.Tensor(mean).view(-1, 4)                   # DBox를 텐서로 변환 torch.Size([8732, 4])
        output.clamp_(max=1, min=0)                                      # DBox가 화상 밖으로 돌출되는 것을 막기 위해 크기를 최소 0, 최대 1로 한다.
        return output

#####################################################################################################
def Detect_inf(loc_data, conf_data, dbox_list):             # SSD 추론 시 conf와 loc의 출력에서 중복을 제거한 BBox 출력
    num_batch = loc_data.size(0)                            # 미니 배치 크기
    num_dbox = loc_data.size(1)                             # DBox의 수 = 8732
    num_classes = conf_data.size(2)                         # 클래스 수 = 21

    softmax = nn.Softmax(dim=-1)
    conf_data = softmax(conf_data)                          # conf_data: [batch_num, 8732, num_classes], 각 디폴트 박스에서 각 클래스에 대한 confidence score가 값으로 들어있음

    output = torch.zeros(num_batch, num_classes, 200, 5)    # 출력 형식을 작성. 텐서 크기 [minibatch 수, 21, 200, 5]

    conf_preds = conf_data.transpose(2, 1)                  # conf_preds: [batch_num, num_classes, 8732]

    ################## 텐서가 autograd 사용중인지 확인 #############        # nm_suppression에서 텐서를 다룰 때 autograd가 켜져 있어 텐서 추적해봄
    # if loc_data.requires_grad:                                        # 추론 시에는 autograd를 사용하지 않는데
    #     print("이 loc_data autograd를 사용합니다.")                      # loc_data가 autograd 사용하는 상태
    # else:
    #     print("이 loc_data autograd를 사용하지 않습니다.")
    # if dbox_list.requires_grad:                                       # dbox_list는 단순 디폴트 박스 생성 함수로 생성된
    #     print("이 dbox_list autograd를 사용합니다.")                     # 텐서라 autograd 사용 안하는 상태
    # else:
    #     print("이 dbox_list autograd를 사용하지 않습니다.")
    #############################################################
    for i in range(num_batch):                                  # 미니 배치 크기 만큼 루프

        decoded_boxes = decode(loc_data[i],dbox_list)           # loc와 DBox로 수정한 BBox [xmin, ymin, xmax, ymax] 를 구한다, torch.Size([8732, 4])

        conf_scores = conf_preds[i].clone()                     # conf의 복사본 작성

        for cl in range(1, num_classes):                        # 클래스별 루프(배경 클래스의 index인 0은 계산하지 않고 index=1부터）

            c_mask = conf_scores[cl].gt(0.01)                   # 임곗값을 넘은 conf의 인덱스를 c_mask로 취득
                                                                # gt는 Greater than을 의미. gt로 임곗값이 넘으면 1, 이하는 0
                                                                # conf_scores:torch.Size([21, 8732])
                                                                # c_mask:torch.Size([8732])

            scores = conf_scores[cl][c_mask]                    # scores는 torch.Size([임곗값을 넘은 BBox 수])

            if scores.nelement() == 0:                          # nelement로 요소 수의 합계를 구함
                continue                                        # 임곗값을 넘은 conf가 없는 경우, 다음 루프

            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)   # c_mask를 decoded_boxes에 적용할 수 있도록 크기 변경
                                                                    # l_mask:torch.Size([8732, 4])
            boxes = decoded_boxes[l_mask].view(-1, 4)               # l_mask를 decoded_boxes로 적용
                                                                    # boxes [선택된 디폴트 박스 수 , 4]
            ######################################################
            # if boxes.requires_grad:                                   # nm_suppression에서 텐서를 다룰 때 autograd가 켜져 있어 텐서 추적해봄
            #     print("이 boxes텐서는 autograd를 사용합니다.")
            # else:
            #     print("이 boxes텐서는 autograd를 사용하지 않습니다.")      # boxes, scores 둘 다 autograd 사용
            # if scores.requires_grad:
            #     print("이 scores텐서는 autograd를 사용합니다.")
            # else:
            #     print("이 scores텐서는 autograd를 사용하지 않습니다.")
            #######################################################

            ids, count = nm_suppression(boxes, scores, 0.45, 200)  # Non-Maximum Suppression을 실시하여 중복되는 BBox 제거
                                                                                # ids：conf의 내림차순으로 Non-Maximum Suppression을 통과한 index 작성
                                                                                # count：Non-Maximum Suppression를 통과한 BBox 수

            output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]),1)  # output에 Non-Maximum Suppression를 뺀 결과 저장

    return output               # torch.Size([1, 21, 200, 5])

#####################################################################################################

class SSD(nn.Module):                           # SSD클래스 작성

    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        self.phase = phase                              # train or inference
        self.num_classes = cfg["num_classes"]           # 클래스 수=1
        self.vgg = make_vgg()                           # SSD 네트워크 작성
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(cfg["num_classes"],cfg["bbox_aspect_num"])  # 'bbox_aspect_num' : [4, 6, 6, 6, 4, 4]
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list().to(device)                               # 디폴트 박스 리스트 생성

    def forward(self, x):
        sources = list()                                # source1 ~ 6 저장할 리스트
        loc = list()                                    # loc의 출력 저장할 리스트
        conf = list()                                   # conf의 출력 저장할 리스트

        for k in range(23):                             # vgg의 conv4_3까지 계산
            x = self.vgg[k](x)
        source1 = self.L2Norm(x)                        # conv4_3의 출력을 L2Norm에 입력하고, source1을 작성하여 sources에 추가
        sources.append(source1)                         # L2Norm하는 이유??, source1만 L2norm 적용하는 이유?

        for k in range(23, len(self.vgg)):              # vgg를 마지막까지 계산하여 source2를 작성하고 sources에 추가
            x = self.vgg[k](x)
        sources.append(x)

        for k, v in enumerate(self.extras):             # extras의 conv와 ReLU 계산
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:                              # conv→ReLU→cov→ReLU를 하여 source에 넣는다.
                sources.append(x)                       # source3～6을 sources에 추가
                                                        # sources [[[512,38,38], [1024,19,19],...[256,1,1]] ...
                                                        #     ...  [[512,38,38], [1024,19,19],...[256,1,1]]]
        ################ sources 차원 확인
        # print('source len', len(sources))               # 6     source
        # print('source len', len(sources[1]))            # 32    batch_num
        # print('source len', len(sources[1][0]))         # 1024  channel
        # print('source len', len(sources[1][0][0]))      # 19    Height
        # print('source len', len(sources[1][0][0][0]))   # 19    Width
        ################

        for (x, l, c) in zip(sources, self.loc, self.conf):         # source1 ~ 6 까지 있어 루프가 6회 실시. loc,conf도 6개의 layers
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())       # source1 ~ 6 에 각각 대응하는 합성곱을 1회씩 적용
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())      # Permute로 요소의 순서를 교체

                                                # l(x), c(x)의 출력 크기는 [batch_num, 4*화면비의 종류 수 , featuremap높이, featuremap폭]
                                                # source에 따라 화면비의 종류 수가 다르며, 번거로워 순서를 바꾸어서 조정
                                                # permute로 [minibatch 수 , featuremap 수 , featuremap 수 ,4*화면비의 종류 수]
                                                # torch.contiguous()은 메모리 상에 연속적으로 요소를 배치하는 명령
                                                # 이후 view 함수 사용하므로 대상의 변수가 메모리 상에 연속적으로 배치되어야 한다.
        #print(loc[0][2])
        #################### loc 차원 확인
        # print('loc len', len(loc))              # 6   source
        # print('loc len', len(loc[0]))           # 32  batch num
        # print('loc len', len(loc[0][0]))        # 38  Height
        # print('loc len', len(loc[0][0][0]))     # 38  Width
        # print('loc len', len(loc[0][0][0][0]))  # 16  aspect ratio 개수 * 좌표 개수
        ##################

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)    # loc의 크기는 torch.Size([batch_num, 34928]) = >
        # (38*38*16) + (19*19*24) + (10*10*24) + (5*5*24) + (3*3*16) + (1*1*16) = 34928

        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)  # conf의 크기는 torch.Size([batch_num, 183372])가 된다 =>
        # (38*38*4*21=121296) + (19*19*6*21=45486) + (10*10*6*21=12600) + (5*5*6*21=3150) + (3*3*4*21=756) + (1*1*4*21=84)
        ################## loc 차원 확인
        # print('loc len', len(loc))        # 32 = 배치 수
        # print('loc len', len(loc[0]))     # 34928 = 피쳐수
        ##################

        loc = loc.view(loc.size(0), -1, 4)                      # loc의 크기는 torch.Size([batch_num, 8732, 4])
        conf = conf.view(conf.size(0), -1, self.num_classes)    # conf의 크기는 torch.Size([batch_num, 8732, 21])

        output = (loc, conf, self.dbox_list)
        if self.phase == "inference":  # 추론 시
            return Detect_inf(output[0], output[1], output[2])  # 반환 값의 크기는 torch.Size([batch_num, 21, 200, 5])

        else:                                                   # 학습 시
            return output                                       # 반환 값은 (loc, conf, dbox_list)의 튜플

#####################################################################################################
def decode(loc, dbox_list):
    """
    오프셋 정보로  DBox를 BBox로 변환한다.
    Parameters
    ----------
    loc:  [8732,4]
        SSD 모델로 추론하는 오프셋 정보
    dbox_list: [8732,4]
        DBox 정보
    Returns
    -------
    boxes : [xmin, ymin, xmax, ymax]
        BBox 정보
    """

    # DBox는 [cx, cy, width, height]로 저장되었다.
    # loc는 [Δcx, Δcy, Δwidth, Δheight]로 저장되었다.
    # (cx, cy) + (Δcx, Δcy) * 0.1 * (w, h)
    # (w, h) *  e^( (Δw, Δh)  * 0.2))               이는 loss에 넣기전 encode의 반대 수식
    boxes = torch.cat((dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:], dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)

    boxes[:, :2] -= boxes[:, 2:] / 2                # 좌표 (xmin,ymin)로 변환
    boxes[:, 2:] += boxes[:, :2]                    # 좌표 (xmax,ymax)로 변환

    return boxes                            # boxes는 예측한 바운딩 박스의 좌표, 크기는 torch.Size([8732, 4])가 된다.

#####################################################################################################

def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    Non-Maximum Suppression을 실시하는 함수
    boxes중 겹치는（overlap이상) BBox 삭제
    Parameters
    ----------
    boxes : [신뢰도 임곗값（0.01)을 넘은 BBox 수,4] 예측한 BBox 정보
    scores :[신뢰도 임곗값（0.01)을 넘은 BBox 수] 예측한 conf 정보, Detect_inf에서 각 클래스마다 호출되어서 1차원이다

    Returns
    -------
    keep : (list) conf의 내림차순으로 nms를 통과한 index를 저장
    count：(int) nms를 통과한 BBox 수
    """
    boxes = boxes.detach()                              # Gradient 계산에서 제외
    scores = scores.detach()

    count = 0                                           # return 값 넣을 변수
    keep = scores.new(scores.size(0)).zero_().long()    # keep：torch.Size([신뢰도 임곗값을 넘은 BBox 수)], 요소는 전부 0

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)                  # 각 BBox의 면적 area 계산

                                                        # boxes 복사. 나중에 BBox 중복도(IOU) 계산 시 모형으로 준비
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    v, idx = scores.sort(0)                             # scores를 오름차순으로 나열

    idx = idx[-top_k:]                          # 상위 top_k개(200개)의 BBox index를 꺼낸다(200개가 존재하지 않는 경우도 있음)

    while idx.numel() > 0:                      # idx의 요소 수가 0이 아닌 한 루프한다.
        i = idx[-1]                             # conf의 최대 index를 i로 지정

                                                # keep의 끝에 conf 최대 index 저장
                                                # 이 index의 BBox와 크게 겹치는 BBox를 삭제
        keep[count] = i
        count += 1
        if idx.size(0) == 1:                    # 마지막 BBox는 루프를 빠져나옴
            break
        idx = idx[:-1]                          # 현재 conf 최대의 index를 keep에 저장했으므로 idx를 하나 감소시킴

        # -------------------
        # 지금부터 keep에 저장한 BBox와 크게 겹치는 BBox를 추출하여 삭제
        # -------------------
        # 하나 감소시킨 idx까지의 BBox를 out으로 지정한 변수로 작성
        # torch.index_select(x1, 0, idx, out=tmp_x1)
        # torch.index_select(y1, 0, idx, out=tmp_y1)
        # torch.index_select(x2, 0, idx, out=tmp_x2)
        # torch.index_select(y2, 0, idx, out=tmp_y2)

        tmp_x1 = torch.index_select(x1, 0, idx)
        tmp_y1 = torch.index_select(y1, 0, idx)
        tmp_x2 = torch.index_select(x2, 0, idx)
        tmp_y2 = torch.index_select(y2, 0, idx)

                                                        # 모든 BBox를 현재 BBox=index가 i로 겹치는 값까지로 설정(clamp)
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

                                                        # w와 h의 텐서 크기를 index 하나 줄인 것으로 한다.
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

                                                        # clamp한 상태에서 BBox의 폭과 높이를 구한다.
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

                                                        # 폭이나 높이가 음수인 것은 0으로 한다.
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

                                                        # clamp된 상태 면적을 구한다.
        inter = tmp_w * tmp_h

                                                        # IoU = intersect 부분 / (area(a) + area(b) - intersect 부분) 계산
        rem_areas = torch.index_select(area, 0, idx)       # 각 BBox의 원래 면적
        union = (rem_areas - inter) + area[i]                   # 두 구역의 합(OR) 면적
        IoU = inter / union
                                                            # IoU가 overlap보다 작은 idx만 남긴다
        idx = idx[IoU.le(overlap)]                          # le은 Less than or Equal to 처리를 하는 연산
                                            # IoU가 overlap보다 큰 idx는 처음 선택한 keep에 저장한 idx와 동일한 물체에 BBox를 둘러싸고 있어 삭제
    return keep, count

#####################################################################################################
# match 함수
def point_form(boxes):                                      # [cx, cy, w, h] => [xmin, ymin, xmax, ymax]
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,boxes[:, :2] + boxes[:, 2:] / 2), 1)

def center_size(boxes):                                     #  [xmin, ymin, xmax, ymax] => [cx, cy, w, h]
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]), 1)  # w, h

def intersect(box_a, box_b):                                # 여러개 박스와 여러개 박스의 겹치는 부분 계산
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B]. # [box_a의 개수, box_b의 개수] (값은 겹치는 부분의 값)
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):                          # 박스의 IOU구함, 여러개와 여러개
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)                          # [n(정답바운딩박스의개수), 8732(디폴트박스의 개수)] (값이 겹치는 부분의 넓이)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union                                    # [A,B]

def match(threshold, truths, dbox, variances, labels, loc_t, conf_t, idx):
    """
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes,        [num_obj, 4].
        dbox: (tensor) Default boxes                [8732 ,4].
        variances:                                  (0.1, 0.2)
        labels: (tensor)                            [num_obj].
        loc_t: (tensor)                             [num_batch, 8732, 21]
        conf_t: (tensor)                            [num_batch, 8732]
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    overlaps = jaccard(truths, point_form(dbox))                            # [obj_num, 8732] 정답과 디폴트 박스간의 IOU값들

    # (Bipartite Matching)

    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)      # overlaps.max(1, keepdim=True)하면 1번째 차원에서 가장 큰 값
                                                                            # best_prior_overlap = [obj_num, 1] 각 정답바운딩박스에 대해 가장 IOU가 높은 디폴트 박스의 IOU
                                                                            # best_prior_idx = [obj_num, 1] 각 정답바운딩박스에 대해 가장 IOU가 높은 디폴트 박스의 index
    ##################Example###############################################################################
    # obj_num가 2일때 overlaps[2, 8732]  = [[1, 3, 4, 5,... 3, 1, 2],      이 줄은 0번째 정답과 각 디폴트 박스간의 IOU
    #                                      [0, 8, 1, 3,... 1, 1, 0]]      이 줄은 1번째 정답과 각 디폴트 박스간의 IOU
    # best_prior_overlap[obj_num, 1]   =    [[5], [8]]
    # best_prior_idx[obj_num, 1]       =    [[3], [1]]
    # 0번째 정답과 3번째 디폴트박스의 IOU가 가장 크고 그 값은 5이다.
    # 1번째 정답과 1번째 디폴트박스의 IOU가 가장 크고 그 값은 8이다.
    #########################################################################################################

    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)      # overlaps.max(0, keepdim=True)하면 0번째 차원에서 가장 큰 값
                                                                            # best_truth_overlap = [1, 8732] 각 디폴트박스에 대해 가장 IOU가 높은 바운딩 박스의 IOU
                                                                            # best_truth_idx = [1, 8732] 각 디폴트박스에 대해 가장 IOU가 높은 바운딩 박스의 index
    ################Example###############################################################################
    # obj_num가 2일때 overlaps[2, 8732] = [[1, 3, 4, 5,... 3, 1, 2],       이 줄은 0번째 정답과 각 디폴트 박스간의 IOU
    #                                     [0, 8, 1, 3,... 1, 0, 0]]       이 줄은 1번째 정답과 각 디폴트 박스간의 IOU
    # best_truth_overlap[1,8732] = [[1, 8, 4, 5,... 3, 1, 2]]
    # best_truth_idx[1,8732]     = [[0, 1, 0, 0,... 0, 0, 0]]
    # 0번째 디폴트 박스와 0번째 정답의 IOU가 가장 크고 그 값은 1이다.
    # 1번째 디폴트 박스와 1번째 정답의 IOU가 가장 크고 그 값은 8이다.
    # 2번째 디폴트 박스와 0번째 정답의 IOU가 가장 크고 그 값은 4이다.
    # 3번째 디폴트 박스와 0번째 정답의 IOU가 가장 크고 그 값은 5이다.
    # ...
    # 8729번째 디폴트 박스와 0번째 정답의 IOU가 가장 크고 그 값은 3이다.
    # 8730번째 디폴트 박스와 0번째 정답의 IOU가 가장 크고 그 값은 1이다.
    # 8731번째 디폴트 박스와 0번째 정답의 IOU가 가장 크고 그 값은 2이다.
    ##########################################################################################################
    best_truth_idx.squeeze_(0)              # [1, 8732] 차원축소 => [8732]
    best_truth_overlap.squeeze_(0)          # [1, 8732] 차원축소 => [8732]
    best_prior_idx.squeeze_(1)              # [obj_num, 1] 차원축소 = > [obj_num]
    best_prior_overlap.squeeze_(1)          # [obj_num, 1] 차원축소 = > [obj_num]

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # 가장 유력한 디폴트 박스의 IOU값을 2로 치환, 0번째 차원에 인덱스에 해당하는 값을 2로 채워라
    #################Example####################
    # overlaps[obj_num, 8732] = [[0.1, 0.4, 0.4,...,0.8, 0.2],
    #                            [0.4, 0.2, 1.7,...,0.9, 0.1]] 일때
    # best_truth_overlap = [0.4, 0.4, 1.7,...,0.9, 0.2]
    # best_prior_idx = [3, 2]
    # 인데
    # best_truth_overlap = [0.4, 0.4, 2.0,...,2.0, 0.2]
    ############################################                # 각 물체 당 하나 이상의 디폴트 박스가 나오게 하기 위해서?
    # TODO refactor: index  best_prior_idx with long tensor
    for j in range(best_prior_idx.size(0)):                     # obj_num 만큼 반복, obj_num가 2이면 j = 0, 1
        best_truth_idx[best_prior_idx[j]] = j                   # 가장 IOU가 높은 디폴트 박스의 순서, 몇번째 디폴트 박스인지
                                                                # best_prior_idx[0] = 0번째 정답과 가장 IOU가 높은 디폴트 박스의 인덱스, 예를 들어 140.
                                                                # best_truth_idx[140] = 140번째 디폴트박스와 가장 IOU가 높은 정답바운딩박스의 인덱스 = 0
                                                                # 여기에 0 다시 대입, 위에 상황아니면 그대로임
    #####################Example#####################
    # 위에 예시 계속해서
    # best_truth_overlap = [0.4, 0.4, 2.0,...,2.0, 0.2] 에서 0.8이 0.9보다 작지만 가로에서 제일 커서 2로됨
    # 하지만 best_truth_overlap에서 바뀐 2.0이 가리키는 obj_index는 1번째 obj를 가리킴
    # 따라서 best_truth_idx를 0으로 바꿔줌

    ############################################
    matches = truths[best_truth_idx]                            #  각 디폴트 박스에서 가장 IOU가 높은 정답의 좌표
                                                                # best_truth_idx = [8732] 안에 값은 obj_num의 인덱스
                                                                # truths = [obj_num, 4]
                                                                # matches = [8732, 4]

    conf = labels[best_truth_idx] + 1                           # Shape: [8732]   각 디폴트 박스에서 가장 IOU가 높은 정답의 클래스
                                                                # best_truth_idx = [8732] 안에 값은 obj_num의 인덱스
                                                                # labels = [obj_num]
                                                                # conf = [8732] 값은 클래스 인덱스

    conf[best_truth_overlap < threshold] = 0                # best_truth_overlap = [8732]이 임계점 안넘으면 conf = [8732]는 0
    loc = encode(matches, dbox, variances)                  # matches는 결국 각 디폴트박스에서 가장 가까운 바운딩박스의 좌표

    loc_t[idx] = loc                            # [8732,4] 디폴트박스와 가장 가까운 바운딩박스의 거리를 encode한 값
    conf_t[idx] = conf                          # [8732] 각 디폴트 박스에서 가장 높은 확률의 클래스

def encode(matched, dbox, variances):
    """
    Args:
        matched:  [8732, 4].  [xmin, ymin, xmax, ymax]  각 디폴트박스에서 가장 가까운 바운딩박스의 좌표
        dbox: [8732,4].        [cx, cy, w, h]           디폴트 박스 리스트
        variances: (1.0, 2.0)
    Return:
        encoded boxes (tensor), Shape: [8732, 4]
    """
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - dbox[:, :2]            # g_cxcy 디폴트박스와 바운딩박스의 중심간의 거리
    #           (xmin, ymin  +  xmax, ymax)  / 2 - cx,cy
    #                   (정답의 cx, cy)           - (디폴트박스의 cx, cy)

    # encode variance => 이걸 왜하는지?
    g_cxcy /= (variances[0] * dbox[:, 2:])
    #       g^cx = (g_cx - d_cx)/(variance * d_w)

    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / dbox[:, 2:]
    #                      정답의 (w, h)      / (디폴트박스의 w, h)

    g_wh = torch.log(g_wh) / variances[1]
    #       g^w = log( g_w / d_w ) / variance

    return torch.cat([g_cxcy, g_wh], 1)             # [8732,4]
                                                                # g_cxcy는 바운딩박스의 중심점과 디폴트박스의 중심점의 거리를 디폴트 박스의 크기 스케일로 나눈것
                                                                # g_wh는 바운딩박스와 디폴트박스간의 크기의 비율을 log스케일로 변환한 것

#####################################################################################################
class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh                        # 0.5 match 함수의 jaccard 계수의 임계치
        self.negpos_ratio = neg_pos                                 # 3:1 Hard Negative Mining의 음과 양 비율
        self.device = device

    def forward(self, predictions, targets):
        """
        손실함수 계산
        Parameters
        ----------
        predictions :   SSD net의 훈련 시 출력 (tuple)
                        (loc=torch.Size([num_batch, 8732, 4]), conf=torch.Size([num_batch,8732, 21]), dbox_list=torch.Size [8732,4])。
        targets :       [num_batch, num_objs, 5]
                        5는 정답인 어노테이션 정보[xmin, ymin, xmax, ymax, label_ind]
        Returns
        -------
        loss_l : 텐서
            loc의 손실 값
        loss_c : 텐서
            conf의 손실 값
        """

        loc_data, conf_data, dbox_lst = predictions         # SSD의 출력이 튜플로 되어 있어 개별적으로 분리함
                                                            # loc_data는 loc_layer를 통과한 출력, 정답 박스 좌표와의 거리(offset) torch.Size([num_batch, 8732, 4])
                                                            # conf_data는 conf_layer를 통과한 출력, 클래스별 확률 torch.Size([num_batch, 8732, 22])
                                                            # dbox_list는 모든 디폴트 박스 좌표, torch.Size [8732,4]
        num_batch = loc_data.size(0)            # 미니 배치 크기
        num_dbox = loc_data.size(1)             # DBox의 수 = 8732
        num_classes = conf_data.size(2)         # 클래스 수= 22
                                                                                      # 손실 계산에 사용할 것을 저장하는 변수 작성
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)    # [배치수, 8732], conf_t_label：각 DBox에 가장 가까운 정답 BBox의 라벨을 저장할 텐서, Longtensor는 int64
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)            # [배치수, 8732, 4], loc_t: 각 DBox에 가장 가까운 정답 BBox의 위치 정보 저장할 텐서

        for idx in range(num_batch):                        # 이미지 한장씩
                                                            # 현재 미니 배치의 정답 어노테이션 BBox와 라벨 취득
            truths = targets[idx][:, :-1].to(self.device)   # BBox [num_objs, [xmin, ymin, xmax, ymax]], [num_objs, 4]

            labels = targets[idx][:, -1].to(self.device)    # [num_objs], 라벨 [물체1 라벨, 물체2 라벨, ...]

            dbox = dbox_lst.to(self.device)                 # dbox = [8732,4], 디폴트 박스를 새로운 변수로 준비

                                                # match 함수를 실행하여 loc_t와 conf_t_label 내용 갱신
                                                # loc_t: 각 DBox에 가장 가까운 정답 BBox 위치 정보가 덮어써짐.(offset)으로
                                                # conf_t_label：각 DBox에 가장 가까운 정답 BBox 라벨이 덮어써짐.
                                                # 단, 가장 가까운 BBox와 jaccard overlap이 0.5보다 작은 경우,
                                                # 정답 BBox의 라벨 conf_t_label은 배경 클래스 0으로 한다.

            variance = [0.1, 0.2]               # 이 variance는 DBox에서 BBox로 보정 계산할 때 사용하는 식의 계수

            match(self.jaccard_thresh, truths, dbox, variance, labels, loc_t, conf_t_label, idx)
                                                # output = loc_t, conf_t_label
                                                # loc_t = 각 dbox에서 가장 가까운 정답 bbox와의 거리 차이(offset) [num_batch, 8732, 21]
                                                # conf_t_label = 각 DBox에 가장 가까운 정답 BBox 라벨 [num_batch, 8732]

                                                            # ----------
                                                            # 위치 손실 : loss_l을 계산
                                                            # Smooth L1 함수로 손실 계산. 단, 물체를 발견한 DBox의 오프셋만 계산
                                                            # ----------
                                                            # 물체를 감지한 BBox를 꺼내는 마스크 작성
        pos_mask = conf_t_label > 0                         # torch.Size([num_batch, 8732])

        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)    # pos_mask를 loc_data 크기로 변형 pos_idx = [num_batch, 8732, 1]

                                                                            # Positive DBox의 loc_data와 지도 데이터 loc_t 취득
        loc_p = loc_data[pos_idx].view(-1, 4)                               # [num_batch * 8732, 4]
        loc_t = loc_t[pos_idx].view(-1, 4)                                  # [num_batch * 8732, 4]

        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')            # 물체를 발견한 Positive DBox의 오프셋 정보 loc_t의 손실(오차)를 계산
                                                                            # loss가 줄어들게 훈련하기 때문에 loc_data는 각 디폴트 박스에서
                                                                            # 가장 가까운 바운딩 박스의 거리를 encode된 값과 가까워지도록 훈련됨
                                                                            # 그래서 추론할때에는 conf가 높은 디폴트박스의 loc값을 encode의 반대로
                                                                            # 계산해야 바운딩 박스 좌표가 나옴

                                                        # ----------
                                                        # 클래스 예측의 손실 : loss_c를 계산
                                                        # 교차 엔트로피 오차 함수로 손실 계산. 단 배경 클래스가 정답인 DBox가 압도적으로 많으므로,
                                                        # Hard Negative Mining을 실시하여 물체 발견 DBox 및 배경 클래스 DBox의 비율이 1:3이 되도록 한다.
                                                        # 배경 클래스 DBox로 예상한 것 중 손실이 적은 것은 클래스 예측 손실에서 제외
                                                        # ----------
        batch_conf = conf_data.view(-1, num_classes)    # [num_batch, 8732, 21] =>  [num_batch * 8732, 21]


        loss_c = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction='none')   # 클래스 예측의 손실함수 계산(reduction='none'으로 하여 합을 취하지 않고 차원 보존)
                                                                                        # conf_t_label.view(-1) = [num_batch * 8732]
                                                                                        # batch_conf = [num_batch * 8732, 21]
                                                                                        # loss_c = [num_batch * 8732]

        # -----------------
        # Negative DBox 중 Hard Negative Mining
        # -----------------
                                                                        # 물체는 label이 1 이상, 라벨 0은 배경을 의미
        num_pos = pos_mask.long().sum(1, keepdim=True)                  # 배치에서 물체를 감지한 디폴트박스의 수, num_pos = [num_batch, 1]
        loss_c = loss_c.view(num_batch, -1)                      # loss_c = torch.Size([num_batch, 8732])
        loss_c[pos_mask] = 0                                            # 물체를 발견한 DBox는 손실 0으로 한다.

                                                                        # 각 DBox 손실의 크기 loss_c 순위 idx_rank를 구함
        _, loss_idx = loss_c.sort(1, descending=True)                   # loss_idx[0][0]은 가장 큰 손실의 디폴트 박스의 loss_c에서의 인덱스
        _, idx_rank = loss_idx.sort(1)                                  # idx_rank[0][0]은 loss_c[0][0]이 loss를 내림차순 했을 때 몇번째인가

        #########################################################################
        #  (주의) 구현된 코드는 특수하며 직관적이지 않음.
        # 위 두 줄의 요점은 각 DBox에 대해 손실 크기가 몇 번째인지의 정보를
        # idx_rank 변수로 빠르게 얻는 코드이다.

        # DBox의 손실 값이 큰 쪽부터 내림차순으로 정렬하여,
        # DBox의 내림차순의 index를 loss_idx에 저장한다.
        # 손실 크기 loss_c의 순위 idx_rank를 구한다.
        # 내림차순이 된 배열 index인 loss_idx를 0부터 8732까지 오름차순으로 다시 정렬하기 위하여
        # 몇 번째 loss_idx의 인덱스를 취할 지 나타내는 것이 idx_rank이다.
        # 예를 들면 idx_rank 요소의 0번째 = idx_rank[0]을 구하는 것은 loss_idx의 값이 0인 요소,
        # 즉 loss_idx[?] =0은 원래 loss_c의 요소 0번째는 내림차순으로 정렬된 loss_idx의
        # 몇 번째입니까? 를구하는 것이 되어 결과적으로,
        # ? = idx_rank[0][0]은 loss_c의 요소 0번째가 내림차순으로 몇 번째인지 나타냄
        ##########################################################################

                                                                        # 배경 DBox의 수 num_neg를 구한다. HardNegative Mining으로
                                                                        # 물체 발견 DBox으 ㅣ수 num_pos의 세 배 (self.negpos_ratio 배)로 한다.
                                                                        # DBox의 수를 초과한 경우에는 DBox의 수를 상한으로 한다.
        num_neg = torch.clamp(num_pos * self.negpos_ratio, max=num_dbox)  # num_neg [num_batch, 1]

                                                                        # idx_rank에 각 DBox의 손실 크기가 위에서부터 몇 번째인지 저장되었다.
                                                                        # 배경 DBox의 수 num_neg보다 순위가 낮은(손실이 큰) DBox를 취하는 마스크 작성
                                                                        # idx_rank [num_batch, 8732]
                                                                        # neg_mask torch.Size([num_batch, 8732])
        neg_mask = idx_rank < num_neg.expand_as(idx_rank)

        # -----------------
        # 지금부터 Negative DBox 중 Hard Negative Mining으로 추출할 것을 구하는 마스크를 작성
        # -----------------

        # 마스크 모양을 고쳐 conf_data에 맞춘다
        # pos_idx_mask는 Positive DBox의 conf를 꺼내는 마스크이다.
        # neg_idx_mask는 Hard Negative Mining으로 추출한 Negative DBox의 conf를 꺼내는 마스크이다.
        # pos_mask：torch.Size([num_batch, 8732])
        # --> pos_idx_mask：torch.Size([num_batch, 8732, 21])
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        # conf_data에서 pos와 neg만 꺼내서 conf_hnm으로 한다.
        # 형태는 torch.Size([num_pos+num_neg, 21])
        conf_hnm = conf_data[(pos_idx_mask + neg_idx_mask).gt(0)].view(-1, num_classes)
        # gt는 greater than (>)의 약칭. mask가 1인 index를 꺼낸다
        # pos_idx_mask+neg_idx_mask는 덧셈이지만 index로 mask를 정리할 뿐임.
        # pos이든 neg이든 마스크가 1인 것을 더해 하나의 리스트로 만들어 이를 gt로 취득한다.

        # 마찬가지로 지도 데이터인 conf_t_label에서 pos와 neg만 꺼내, conf_t_label_hnm 으로
        # torch.Size([pos+neg]) 형태가 된다
        conf_t_label_hnm = conf_t_label[(pos_mask + neg_mask).gt(0)]

        # confidence의 손실함수 계산(요소의 합계=sum을 구함)
        # 수정 'mean'으로
        # loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')

        # 물체를 발견한 BBox의 수 N (전체 미니 배치의 합계) 으로 손실을 나눈다.
        # 수정 num_pos로 loss_c를 나누는게 아니라 num_pos+num_neg으로 나눠야 하는거 아닌가
        # N = num_pos.sum()
        # loss_l /= N
        # loss_c /= N

        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='mean')
        #print("loss_l", loss_l)
        N = num_pos.sum() + 0.01
        loss_l /= N

        return loss_l, loss_c

#####################################################################################################

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):           # 학습 및 검증 실시

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')         # GPU 사용 확인
    print(f'사용 중인 장치 : {device}')
    net.to(device)                                                                  # 네트워크를 gpu로
    torch.backends.cudnn.benchmark = True                                           # 네트워크가 어느정도 고정되면 고속화

    iteration = 1
    epoch_train_loss = 0.0                              # 에포크 손실 합
    epoch_val_loss = 0.0
    logs = []

    for epoch in range(num_epochs + 1):

        t_epoch_start = time.time()                     # 시작 시간 저장
        t_iter_start = time.time()

        print('-' * 20)
        print(f'Epoch : {epoch + 1}/{num_epochs}')
        print('-' * 20)

        for phase in ['train', 'val']:                  # 에폭 별 훈련 및 검증 루프
            if phase == 'train':
                net.train()                             # 모델을 훈련 모드로
                print('(train)')
            else:
                if ((epoch + 1) % 10 == 0):             # 검증은 10 회 중 1회만 실시
                    net.eval()                          # 모델을 검증 모드로
                    print('-' * 20)
                    print('(val)')
                else:
                    continue

            for images, targets in dataloaders_dict[phase]:         # 데이터 로더에서 미니 배치씩 꺼내 루프
                images = images.to(device)                          # torch.Size([32, 3, 300, 300])
                targets = [ann.to(device) for ann in targets]       # 텐서가 아니라 하나씩 장치로 이동 [batch_num, obj_num, 5(좌표, 클래스인덱스)]

                optimizer.zero_grad()                               # 옵티마이저 초기화

                with torch.set_grad_enabled(phase == 'train'):      # 훈련모드면 기울기 계산 활성화, 추론이면 비활성화
                    outputs = net(images)                           # train모드로 모델 통과 후 출력, (loc, conf, dbox_list)의 튜플
                    loss_l, loss_c = criterion(outputs, targets)    # criterion = MultiBoxLoss()
                    loss = loss_l + loss_c                          # loss = <class 'torch.Tensor'>
                    if phase == 'train':                            # 훈련 시에는 역전파
                        loss.backward()                             # 자동미분 활용하여 손실의 기울기 계산
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0) # 경사가 너무 커지면 계산이 부정확해 clip에서 최대경사 2.0에 고정
                        optimizer.step()                            # 기울기를 사용하여 가중치 업데이트

                        if (iteration % 10 == 0):                   # 10 iter에 한 번 손실 표시
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print(f'반복 {iteration} || Loss: {loss.item(): .4f} | | 10 iter: {duration: .4f} \ sec.')
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:                                           # 검증 시
                        epoch_val_loss += loss.item()

            t_epoch_finish = time.time()
            print('-' * 20)
            print(
                f'epoch {epoch + 1} || Epoch_Train_loss : {epoch_train_loss:.4f} || Epoch_val_loss: {epoch_val_loss: .4f}')
            print(f'timer : {t_epoch_finish - t_epoch_start:.4f} sec')

            # 로그 저장
            los_epoch = {'epoch': epoch + 1, 'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
            logs.append(los_epoch)
            df = pd.DataFrame(logs)
            df.to_csv('log_output.csv')

            epoch_train_loss = 0.0
            epoch_val_loss = 0.0

            # 네트워크 저장
            if ((epoch + 1) % 10 == 0):
                torch.save(net.state_dict(), 'weights/ssd300_class22_0913_' + str(epoch + 1) + '.pth')

#####################################################################################################

#####################################################################################################

# 화상 예측
class SSDPredictShow():

    def __init__(self, eval_categories, net):
        self.eval_categories = eval_categories
        self.net = net

        color_mean = (128, 128, 128)
        input_size = 300
        self.transform = DataTransform(input_size, color_mean)

    def show(self, image_file_path, data_confidence_level):

        rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(image_file_path, data_confidence_level)

        self.vis_bbox(rgb_img, bbox=predict_bbox, label_index=pre_dict_label_index,scores=scores, label_names=self.eval_categories)

    def ssd_predict(self, image_file_path, data_confidence_level=0.7):
        """
        SSD 예측 후 예측 좌표, 예측 클래스, conf score 반환
        Parameters
        ----------
        image_file_path:  str
        data_confidence_level: float

        Returns
        -------
        rgb_img, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        img = cv2.imread(image_file_path)
        height, width, channels = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        phase = "val"
        img_transformed, boxes, labels = self.transform(img, phase, "", "")
        img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        img = img.to(device)
        self.net.eval()
        x = img.unsqueeze(0)  # torch.Size([1, 3, 300, 300]) 배치 차원 맞춰주기

        detections = self.net(x)  # detections torch.Size([1, 21, 200, 5])  200은 클래스 별 nms 거친 conf score 높은 예측 박스 순

        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections.cpu().detach().numpy()
        find_index = np.where(detections[:, 0:, :,0] >= data_confidence_level)
        # detections[:, 0:, :, 0] = torch.Size([1, 21, 200]) 값은 conf_score
        # find_index 튜플(array[신뢰도가 임계값 넘은 예측 박스의수], array[신뢰도가 임계값 넘은 예측 박스의수],array[신뢰도가 임계값 넘은 예측 박스의수])
        # 첫 번째 array는 batch index 값
        # 두 번째 array는 class index 값
        # 세 번째 array는 예측 bbox index 값
        # find_index가 ( [0, 1], [3, 13], [5,2])이면
        # 신뢰도가 임계값 넘은 예측 박스의 수가 2개이고,그 박스는
        # detections[0][3][5] 와 detections[1][13][2] 이 두개를 가리킨다

        detections = detections[find_index]  # detections = (conf_level넘은 예측 박스의 수, 5) 값은 conf+loc = 5

        # print(len(find_index[0]))
        # print(len(find_index[1]))     # 셋 동일
        # print(len(find_index[2]))

        for i in range(len(find_index[1])):                                 # len(find_index[1]) = 신뢰도 넘는 예측 박스의 수
            if (find_index[1][i]) > 0:                                      # find_index[1][i] = 신뢰도 넘는 예측 박스들의 예측 클래스 즉, 배경이 아니라 물체로 예측했으면
                sc = detections[i][0]                                       # detections[i][0] = 신뢰도 넘는 예측 박스들의 conf score
                bbox = detections[i][1:] * [width, height, width,height]    # detections[i][1:] = 신뢰도 넘는 예측 박스들의 loc[xmin, ymin, xmax, ymax](0~1) => (0~원본이미지 높이 넓이)
                                                                            # 여기서 loc값은 모델 거친 로짓 그대로가 아니라 Detect에서 decode거친 실제 좌표값(0~1) 임
                label_ind = find_index[1][i] - 1  # 박스 별 예측 클래스에 -1 하여 cfg 딕셔너리 클래스와 맞춤

                predict_bbox.append(bbox)                               # 예측 좌표(0~원본이미지 높이 넓이)
                pre_dict_label_index.append(label_ind)                  # 예측 클래스 인덱스
                scores.append(sc)                                       # confidence score

        return rgb_img, predict_bbox, pre_dict_label_index, scores

    def vis_bbox(self, rgb_img, bbox, label_index, scores, label_names):
        """
        예측 좌표, 클래스, 스코어를 통해 이미지에 바운딩박스 출력
        Parameters
        ----------
        rgb_img:rgb 원본
        bbox:           list [예측 물체의 수, 4]  예측 바운딩박스 좌표
        label_index:    list[예측 물체의수]       예측 클래스 인덱스
        scores:         list [예측 물체의 수]     스코어
        label_names:    list [예측 물체의 수]     예측 클래스 이름

        Returns
        -------
        rgb_img와 예측한 박스 그리기
        """
        num_classes = len(label_names)
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()  # 클래스 마다 다른색 부여

        #plt.figure(figsize=(10, 10))
        ax.imshow(rgb_img)
        #currentAxis = ax.gca()

        for i, bb in enumerate(bbox):  # 예측한 물체의 수만큼 반복, bb = 예측 좌표
            label_name = label_names[label_index[i]]  # 예측한 클래스 이름
            color = colors[label_index[i]]

            if scores is not None:
                sc = scores[i]
                display_txt = '%s: %.2f' % (label_name, sc)
            else:
                display_txt = '%s: ans' % label_name

            xy = (bb[0], bb[1])  # 원본에서의 시작좌표(x, y) (0 ~ 원본 크기)
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]

            ax.add_patch(plt.Rectangle(xy, width, height, fill=False, edgecolor=color, linewidth=2))
            ax.text(xy[0], xy[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})

        #plt.show()

#####################################################################################################
######  main    ######
#####################################################################################################

torch.manual_seed(1234)                 # 구현 결과 일정하게 하기위한 시드 고정
np.random.seed(1234)
random.seed(1234)

rootpath = './우산1000/'

voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'umbrella']

ssd_cfg = {                                                         # configure 딕셔너리
    'num_classes': 22,                                              # 클래스 수
    'num_epochs': 100,                                              # 에포크 수
    'input_size': 300,                                              # 입력 이미지 사이즈 : 300 x 300
    'batch_size': 32,                                               # 배치 사이즈
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],                          # 출력할 DBox 화면비 종류의 개수
    'feature_maps': [38, 19, 10, 5, 3, 1],                          # 각 source(추출한 feature map)의 크기
    'scale': [0.2, 0.34, 0.48, 0.62, 0.76, 0.9, 1.04],              # 각 source에서 작은 1:1 defalutBox의 스케일
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],       # 1,5,6번째 feature map에서는 1:1, 1:2, 2:1, 또 다른 1:1 총 4개
                                                                    # 2,3,4에서는 1:1, 1:2, 2:1, 1:3, 3:1, 또 다른 1:1 총 6개이다
                                                                    # 작은 dbox가 (w, h)일때 1:2인 dbox는 (w, 2h)가 아니라 (w/sqrt(2), h*sqrt(2))
                                                                    # class DBox의  make_dbox_list참조
}

train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)
                                                    # 저장된 데이터 파일로부터 train, validation의 이미지와 어노테이션 파일 경로를 리스트로 저장
                                                    # <class 'list'> 이미지와 어노테이션 파일 경로가 string으로 저장
                                                    # ['~.jpg', '~.jpg', '~.jpg', ... '~.jpg', '~.jpg']
                                                    # ['~.xml', '~.xml', '~.xml', ... '~.xml', '~.xml']

color_mean = (128, 128, 128)                        # BGR 색 평균 값

train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train',              # 데이터셋 객체 생성, 해당 파일들과 DataTransform에 대한 정보만 저장
                           transform=DataTransform(ssd_cfg['input_size'], color_mean),  # 아래의 로더에 들어갈 때 __getitem__ 메소드가 호출 되면서 전처리 수행
                           transform_anno=Anno_xml2list(voc_classes))
val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val',                      # print(train_dataset[0]) 하면 __getitem__이 호출되면서 [img, gt, w, h] 출력,
                         transform=DataTransform(ssd_cfg['input_size'], color_mean),    # gt는 어노테이션[xmin, ymin, xmax, ymax, label]
                         transform_anno=Anno_xml2list(voc_classes))

train_dataloader = data.DataLoader(train_dataset, batch_size=ssd_cfg['batch_size'], shuffle=True,
                                   collate_fn=od_collate_fn)                                        # dataloader 작성
val_dataloader = data.DataLoader(val_dataset, batch_size=ssd_cfg['batch_size'], shuffle=False,
                                 collate_fn=od_collate_fn)                                          # 생성한 데이터셋으로 데이터로더 생성, 데이터셋의 전처리 수행 후 배치단위로 저장

dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}                               # 사전 오브젝트로 정리

################# 데이터 로더 출력해보기 #########
# batch_iterator = iter(dataloaders_dict['train'])            # 반복자로 변환 , 데이터로더는 []인덱싱으로 접근안됨, iter나 for문 등의 반복자로 접근 가능
# images, targets = next(batch_iterator)                      # 첫번째 요소 추출, images는 텐서, targets은 리스트, targets[0]은 텐서
# print('첫번째 데이터로더의 이미지크기', images.size())           # torch.Size([32(batch_size), 3, 300, 300])
# print('첫번째 데이터로더의 targets 크기', targets[0].size())    # torch.Size([2(한 이미지의 물체 수), 5(xmin, ymin, xmax, ymax, label)])
##############################################

net = SSD(phase='train', cfg=ssd_cfg)  # 신경망 구현, SSD300 설정
'''
pretrained_vgg16 = models.vgg16(pretrained=True)            # torchvision.models.VGG16 => torchvision의 훈련된 VGG16모델

pretrained_vgg16_weights = pretrained_vgg16.state_dict()    # VGG16의 가중치 저장

net_vgg_weights = net.vgg.state_dict()                      # SSD모델의 VGG16 부분의 가중치 저장

                                                                                # VGG16의 가중치와 SSD 모델의 VGG16 부분의 가중치가 일치하는지 확인 후 로드
                                                                                # 이 단계는 각 모델의 레이어가 일치하도록 확인하는 것이 중요합니다.
                                                                                # 각 레이어의 이름이 달라 직접 대입

net_vgg_weights['0.weight'] = pretrained_vgg16_weights['features.0.weight']
net_vgg_weights['0.bias'] = pretrained_vgg16_weights['features.0.bias']
net_vgg_weights['2.weight'] = pretrained_vgg16_weights['features.2.weight']
net_vgg_weights['2.bias'] = pretrained_vgg16_weights['features.2.bias']
net_vgg_weights['5.weight'] = pretrained_vgg16_weights['features.5.weight']
net_vgg_weights['5.bias'] = pretrained_vgg16_weights['features.5.bias']
net_vgg_weights['7.weight'] = pretrained_vgg16_weights['features.7.weight']
net_vgg_weights['7.bias'] = pretrained_vgg16_weights['features.7.bias']
net_vgg_weights['10.weight'] = pretrained_vgg16_weights['features.10.weight']
net_vgg_weights['10.bias'] = pretrained_vgg16_weights['features.10.bias']
net_vgg_weights['12.weight'] = pretrained_vgg16_weights['features.12.weight']
net_vgg_weights['12.bias'] = pretrained_vgg16_weights['features.12.bias']
net_vgg_weights['14.weight'] = pretrained_vgg16_weights['features.14.weight']
net_vgg_weights['14.bias'] = pretrained_vgg16_weights['features.14.bias']
net_vgg_weights['17.weight'] = pretrained_vgg16_weights['features.17.weight']
net_vgg_weights['17.bias'] = pretrained_vgg16_weights['features.17.bias']
net_vgg_weights['19.weight'] = pretrained_vgg16_weights['features.19.weight']
net_vgg_weights['19.bias'] = pretrained_vgg16_weights['features.19.bias']
net_vgg_weights['21.weight'] = pretrained_vgg16_weights['features.21.weight']
net_vgg_weights['21.bias'] = pretrained_vgg16_weights['features.21.bias']
net_vgg_weights['24.weight'] = pretrained_vgg16_weights['features.24.weight']
net_vgg_weights['24.bias'] = pretrained_vgg16_weights['features.24.bias']
net_vgg_weights['26.weight'] = pretrained_vgg16_weights['features.26.weight']
net_vgg_weights['26.bias'] = pretrained_vgg16_weights['features.26.bias']
net_vgg_weights['28.weight'] = pretrained_vgg16_weights['features.28.weight']
net_vgg_weights['28.bias'] = pretrained_vgg16_weights['features.28.bias']

net.vgg.load_state_dict(net_vgg_weights)            # 업데이트된 가중치를 SSD 모델의 VGG16 부분에 로드합니다.
'''
# 만약 SSD모델 전체의 가중치를 가져오고 싶다면 아래 코드

# ssd_model = ssd300_vgg16(pretrained=True)  # torchvision.models.detection.ssd300_vgg16 => torchvision의 훈련된 SSD모델
# net_state = net.state_dict()
# ssd_model_state = ssd_model.state_dict()

########################
# print('net_state', net_state.keys())           # 작성한 모델 레이어 확인
# print('model_state', ssd_model_state.keys())   # 불러온 모델 레이어 확인
########################
'''
net_state['vgg.0.weight'] = ssd_model_state['backbone.features.0.weight']
net_state['vgg.0.bias'] = ssd_model_state['backbone.features.0.bias']
net_state['vgg.2.weight'] = ssd_model_state['backbone.features.2.weight']
net_state['vgg.2.bias'] = ssd_model_state['backbone.features.2.bias']
net_state['vgg.5.weight'] = ssd_model_state['backbone.features.5.weight']
net_state['vgg.5.bias'] = ssd_model_state['backbone.features.5.bias']
net_state['vgg.7.weight'] = ssd_model_state['backbone.features.7.weight']
net_state['vgg.7.bias'] = ssd_model_state['backbone.features.7.bias']
net_state['vgg.10.weight'] = ssd_model_state['backbone.features.10.weight']
net_state['vgg.10.bias'] = ssd_model_state['backbone.features.10.bias']
net_state['vgg.12.weight'] = ssd_model_state['backbone.features.12.weight']
net_state['vgg.12.bias'] = ssd_model_state['backbone.features.12.bias']
net_state['vgg.14.weight'] = ssd_model_state['backbone.features.14.weight']
net_state['vgg.14.bias'] = ssd_model_state['backbone.features.14.bias']
net_state['vgg.17.weight'] = ssd_model_state['backbone.features.17.weight']
net_state['vgg.17.bias'] = ssd_model_state['backbone.features.17.bias']
net_state['vgg.19.weight'] = ssd_model_state['backbone.features.19.weight']
net_state['vgg.19.bias'] = ssd_model_state['backbone.features.19.bias']
net_state['vgg.21.weight'] = ssd_model_state['backbone.features.21.weight']
net_state['vgg.21.bias'] = ssd_model_state['backbone.features.21.bias']

net_state['vgg.24.weight'] = ssd_model_state['backbone.extra.0.1.weight']
net_state['vgg.24.bias'] = ssd_model_state['backbone.extra.0.1.bias']
net_state['vgg.26.weight'] = ssd_model_state['backbone.extra.0.3.weight']
net_state['vgg.26.bias'] = ssd_model_state['backbone.extra.0.3.bias']
net_state['vgg.28.weight'] = ssd_model_state['backbone.extra.0.5.weight']
net_state['vgg.28.bias'] = ssd_model_state['backbone.extra.0.5.bias']
net_state['vgg.31.weight'] = ssd_model_state['backbone.extra.0.7.1.weight']
net_state['vgg.31.bias'] = ssd_model_state['backbone.extra.0.7.1.bias']
net_state['vgg.33.weight'] = ssd_model_state['backbone.extra.0.7.3.weight']
net_state['vgg.33.bias'] = ssd_model_state['backbone.extra.0.7.3.bias']

net_state['extras.0.weight'] = ssd_model_state['backbone.extra.1.0.weight']
net_state['extras.0.bias'] = ssd_model_state['backbone.extra.1.0.bias']
net_state['extras.1.weight'] = ssd_model_state['backbone.extra.1.2.weight']
net_state['extras.1.bias'] = ssd_model_state['backbone.extra.1.2.bias']
net_state['extras.2.weight'] = ssd_model_state['backbone.extra.2.0.weight']
net_state['extras.2.bias'] = ssd_model_state['backbone.extra.2.0.bias']
net_state['extras.3.weight'] = ssd_model_state['backbone.extra.2.2.weight']
net_state['extras.3.bias'] = ssd_model_state['backbone.extra.2.2.bias']
net_state['extras.4.weight'] = ssd_model_state['backbone.extra.3.0.weight']
net_state['extras.4.bias'] = ssd_model_state['backbone.extra.3.0.bias']
net_state['extras.5.weight'] = ssd_model_state['backbone.extra.3.2.weight']
net_state['extras.5.bias'] = ssd_model_state['backbone.extra.3.2.bias']
net_state['extras.6.weight'] = ssd_model_state['backbone.extra.4.0.weight']
net_state['extras.6.bias'] = ssd_model_state['backbone.extra.4.0.bias']
net_state['extras.7.weight'] = ssd_model_state['backbone.extra.4.2.weight']
net_state['extras.7.bias'] = ssd_model_state['backbone.extra.4.2.bias']

net_state['loc.0.weight'] = ssd_model_state['head.regression_head.module_list.0.weight']
net_state['loc.0.bias'] = ssd_model_state['head.regression_head.module_list.0.bias']
net_state['loc.1.weight'] = ssd_model_state['head.regression_head.module_list.1.weight']
net_state['loc.1.bias'] = ssd_model_state['head.regression_head.module_list.1.bias']
net_state['loc.2.weight'] = ssd_model_state['head.regression_head.module_list.2.weight']
net_state['loc.2.bias'] = ssd_model_state['head.regression_head.module_list.2.bias']
net_state['loc.3.weight'] = ssd_model_state['head.regression_head.module_list.3.weight']
net_state['loc.3.bias'] = ssd_model_state['head.regression_head.module_list.3.bias']
net_state['loc.4.weight'] = ssd_model_state['head.regression_head.module_list.4.weight']
net_state['loc.4.bias'] = ssd_model_state['head.regression_head.module_list.4.bias']
net_state['loc.5.weight'] = ssd_model_state['head.regression_head.module_list.5.weight']
net_state['loc.5.bias'] = ssd_model_state['head.regression_head.module_list.5.bias']
'''

'''
net_state['conf.0.weight'] = ssd_model_state['head.classification_head.module_list.0.weight']       # 불러온 모델이 다른 클래스 수로 훈련되어 있어서 conf레이어의 가중치 수가 다름
net_state['conf.0.bias'] = ssd_model_state['head.classification_head.module_list.0.bias']
net_state['conf.1.weight'] = ssd_model_state['head.classification_head.module_list.1.weight']
net_state['conf.1.bias'] = ssd_model_state['head.classification_head.module_list.1.bias']
net_state['conf.2.weight'] = ssd_model_state['head.classification_head.module_list.2.weight']
net_state['conf.2.bias'] = ssd_model_state['head.classification_head.module_list.2.bias']
net_state['conf.3.weight'] = ssd_model_state['head.classification_head.module_list.3.weight']
net_state['conf.3.bias'] = ssd_model_state['head.classification_head.module_list.3.bias']
net_state['conf.4.weight'] = ssd_model_state['head.classification_head.module_list.4.weight']
net_state['conf.4.bias'] = ssd_model_state['head.classification_head.module_list.4.bias']
net_state['conf.5.weight'] = ssd_model_state['head.classification_head.module_list.5.weight']
net_state['conf.5.bias'] = ssd_model_state['head.classification_head.module_list.5.bias']
'''
net_weights = torch.load('./weights/ssd300_loss_sum_50.pth', map_location={'cuda:0': 'cpu'})

net_state = net.state_dict()

for name, param in net_weights.items():                 # 훈련된 모델이 21개 클래스였어서 가중치 개수가 안맞음
    if 'conf' in name:                                  # 그래서 conf레이어 제외하고 가중치 업데이트
        continue
    else:
        net_state[name].copy_(param)

net.load_state_dict(net_state)                          # 로드할때 레이어 키 일치하는지 확인

def weights_init(m):                                    # ssd의 기타 네트워크 가중치는 He의 초기치로 초기화
    if isinstance(m, nn.Conv2d):                        # layer가 conv이면
        init.kaiming_normal_(m.weight.data)             # m.wight.data는 m레이어의 가중치 텐서, torch.nn.init
        if m.bias is not None:                          # bias 항이 있는 경우
            init.constant_(m.bias, 0.0)             # m.bias는 m레이어의 바이어스 텐서

# net.extras.apply(weights_init)                        # He의 초기치 적용, apply를 활용하면 각 레이어에 한번씩 초기화 함수 적용
# net.loc.apply(weights_init)
net.conf.apply(weights_init)                            # conf를 제외한 레이어는 가중치 불러와서 conf만 가중치 초기화

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')             # GPU를 사용할 수 있는지 확인
print(f'사용 중인 장치 : {device}')
print('네트워크 설정 완료 : 학습된 가중치를 로드했습니다.')

criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)              # 손실함수 설정

optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)   # 최적화 기법 선정

train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=ssd_cfg['num_epochs']) # 학습 및 검증 실시

#####################################################################################################
# 추론
#####################################################################################################

net = SSD(phase='inference', cfg=ssd_cfg)                       # SSD 추론 모드
net.to(device)
net_weights = torch.load('./weights/ssd300_class22_0913_100.pth', map_location={'cuda:0': 'cpu'})  # SSD의 학습된 가중치 불러오기

net.load_state_dict(net_weights)                                # 학습된 가중치 로드
print('네트워크 설정 완료 : 학습 가중치를 로드했습니다.')

# img_file_path = './crawled_img/dog_true/강아지_75.jpg'
#img_file_path = './우산1000/우산_train/우산_121.jpg'

img_path = []

img_path.append('./우산1000/우산_test/우산_5605.jpg')
img_path.append('./우산1000/우산_test/우산_5601.jpg')
img_path.append('./우산1000/우산_test/우산_5625.jpg')
img_path.append('./우산1000/우산_test/우산_5620.jpg')
img_path.append('./우산1000/우산_test/우산_5651.jpg')
img_path.append('./우산1000/우산_test/우산_5657.jpg')
img_path.append('./우산1000/우산_test/우산_5733.jpg')
img_path.append('./우산1000/우산_test/우산_5776.jpg')
img_path.append('./우산1000/우산_test/우산_5795.jpg')
img_path.append('./우산1000/우산_test/우산_5849.jpg')
img_path.append('./우산1000/우산_test/우산_5858.jpg')
img_path.append('./우산1000/우산_test/우산_5910.jpg')
img_path.append('./우산1000/우산_test/우산_5962.jpg')
img_path.append('./우산1000/우산_test/우산_6130.jpg')
img_path.append('./우산1000/우산_test/우산_5912.jpg')
img_path.append('./우산1000/우산_test/우산_6125.jpg')
img_path.append('./우산1000/우산_test/우산_6122.jpg')
img_path.append('./우산1000/우산_test/우산_6195.jpg')
img_path.append('./우산1000/우산_test/우산_6226.jpg')
img_path.append('./우산1000/우산_test/우산_6210.jpg')


ssd = SSDPredictShow(eval_categories=voc_classes, net=net)              # 예측 하고 결과 출력하기 위한 클래스
fig, axs = plt.subplots(4, 5, figsize=(12, 10))             # 여러 이미지 출력하기

for imgpath, ax in zip(img_path, axs.flatten()):
    ssd.show(imgpath, data_confidence_level=0.7)                        # 이미지 예측 후 결과 출력

plt.tight_layout()
plt.show()

#####################################################################################################
# mAP
#####################################################################################################
def ssd_predict(image_file_path, data_confidence_level=0.5):

    img = cv2.imread(image_file_path)
    height, width, channels = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    phase = "val"
    transform = DataTransform(ssd_cfg['input_size'], color_mean)
    img_transformed, boxes, labels = transform(img, phase, "", "")
    img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = img.to(device)
    net.eval()
    x = img.unsqueeze(0)  # torch.Size([1, 3, 300, 300]) 배치 차원 맞춰주기

    detections = net(x)  # detections torch.Size([1, 22, 200, 5])  200은 클래스 별 nms 거친 conf score 높은 예측 박스 순

    predict_bbox = []
    pre_dict_label_index = []
    scores = []
    detections = detections.cpu().detach().numpy()
    find_index = np.where(detections[:, 0:, :,0] >= data_confidence_level)  # detections[:, 0:, :, 0] = torch.Size([1, 21, 200]) 값은 conf_score
    # find_index 튜플(array[신뢰도가 임계값 넘은 예측 박스의수], array[신뢰도가 임계값 넘은 예측 박스의수],array[신뢰도가 임계값 넘은 예측 박스의수])
    # 첫 번째 array는 batch index 값
    # 두 번째 array는 class index 값
    # 세 번째 array는 예측 bbox index 값
    # find_index가 ( [0, 1], [3, 13], [5,2])이면
    # 신뢰도가 임계값 넘은 예측 박스의 수가 2개이고,그 박스는
    # detections[0][3][5] 와 detections[1][13][2] 이 두개를 가리킨다

    detections = detections[find_index]  # detections = (conf_level넘은 예측 박스의 수, 5) 값은 conf+loc = 5

    # print(len(find_index[0]))
    # print(len(find_index[1]))     # 셋 동일
    # print(len(find_index[2]))
    prediction_list = []
    for i in range(len(find_index[1])):  # len(find_index[1]) = 신뢰도 넘는 예측 박스의 수
        if (find_index[1][i]) > 0:  # find_index[1][i] = 신뢰도 넘는 예측 박스들의 예측 클래스 즉, 배경이 아니라 물체로 예측했으면
            prediction = []
            sc = detections[i][0]  # detections[i][0] = 신뢰도 넘는 예측 박스들의 conf score
            bbox = detections[i][1:] * [width, height, width,
                                        height]  # detections[i][1:] = 신뢰도 넘는 예측 박스들의 loc[xmin, ymin, xmax, ymax](0~1) => (0~원본이미지 높이 넓이)
            # 여기서 loc값은 모델 거친 로짓 그대로가 아니라 Detect에서 decode거친 실제 좌표값(0~1) 임
            label_ind = find_index[1][i] - 1  # 박스 별 예측 클래스에 -1 하여 cfg 딕셔너리 클래스와 맞춤

            prediction.append(bbox)  # 예측 좌표(0~원본이미지 높이 넓이)
            prediction.append(label_ind)  # 예측 클래스 인덱스
            prediction.append(sc)  # confidence score
            prediction_list.append(prediction)


    return prediction_list  #  [ [[xmin, ymin, xmax, ymax], class_idx, conf_score], .. ,[]]
                            # [바운딩박스 개수][3(좌표, 인덱스, 신뢰도)]

def make_anno_list(txt_path, width, height, cfg_classes):

    img_anno = []  # 한 이미지 내 모든 물체의 어노테이션을 이 리스트에 저장,

    if 'txt' in txt_path:
        with open(txt_path, 'r') as file:
            for line in file:
                anno = []
                parts = line.strip().split()
                cx, cy, w, h = map(float, parts[1:])
                xmin = (cx - (w/2)) * width
                xmax = (cx + (w/2)) * width
                ymin = (cy - (h/2)) * height
                ymax = (cy + (h / 2)) * height
                label = int(parts[0])+20
                anno.append([xmin, ymin, xmax, ymax])
                anno.append(label)
                img_anno.append(anno)

        return img_anno  # [[[xmin, ymin, xmax, ymax], label_ind], ... ]

    else:
        xml = ET.parse(txt_path).getroot()  # xml파일을 파싱해서 ElementTree 객체를 생성하고 루트 요소 반환

        for obj in xml.iter('object'):  # 이미지 내 물체의 수 만큼 반복
            difficult = int(obj.find('difficult').text)  # annotation에서 difficult가 1로 설정된 것은 제외
            if difficult == 1:
                continue

            obj_anno = []  # 한 물체의 어노테이션을 저장하는 리스트
            anno = []
            name = obj.find('name').text.lower().strip()  # 물체 이름
            bbox = obj.find('bndbox')  # 바운딩 박스 정보

            b_pts = ['xmin', 'ymin', 'xmax', 'ymax']  # for문 활용해 찾을 어노테이션 정보 저장 리스트

            for b_pt in (b_pts):  # 어노테이션의 xmin, ymin, xmax, ymax를 취득하고, 0 ~ 1로 규격화,['xmin', 'ymin', 'xmax', 'ymax']
                cur_pixel = int(bbox.find(b_pt).text) - 1  # VOC는 원점이 (1, 1)이므로 1을 빼서 (0, 0)으로 한다.

                # if b_pt == 'xmin' or b_pt == 'xmax':  # 폭, 높이로 규격화
                #     cur_pixel /= width  # x 방향의 경우 폭으로 나눔 => 0~1의 좌표
                # else:  # y 방향의 경우 높이로 나눔 => 0~1의 좌표
                #     cur_pixel /= height  # 이미지 넓이가 100이고 xmin이 30이면 0.3으로 정규화

                obj_anno.append(cur_pixel)  # xmin, ymin, xmax, ymax 순서로 입력

            label_idx = cfg_classes.index(name)  # 어노테이션 클래스명 index를 취득하여 추가


            anno += [obj_anno]  # [xmin, ymin, xmax, ymax, label_ind]를 더한다.
            anno.append(label_idx)
            img_anno.append(anno)

        return img_anno  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

def compute_iou(pred_box, gt_box):
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[2])
    y2 = min(pred_box[3], gt_box[3])

    inter_area =  max(0, x2 - x1) * max(0, y2 - y1)
    pred_box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    union_area = pred_box_area + gt_box_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def ElevenPointInterpolatedAP(rec, prec):
    recall_range = np.linspace(0, 1, 11)
    recall_range = list(recall_range[::-1])                 # recallValues = [1.0, 0.9, ..., 0.0]
    interpol_point =  []
    #print('recall len',rec)
    for r in recall_range:
        argGreaterRecalls = np.argwhere(rec[:] >= r)       # argGreaterRecalls = r보다 큰 recall값을 가지는 인덱스 리스트
        pmax = 0
        #print('argGreaterRecalls', r, argGreaterRecalls)

        if argGreaterRecalls.size != 0:
            pmax = max(prec[argGreaterRecalls.min():])      # r 구간에서, 즉 현재 recall보다 큰 recall의 precision중 가장 큰 값
        #print('pmax', pmax)

        interpol_point.append(pmax)
        #print('interpol_point', interpol_point)

    ap = sum(interpol_point) / 11

    return ap

def AP(predictions, groundtruths, iou_threshold = 0.5):
                                            # predictions = [모든 이미지의 예측 박스의 수][4(좌표, 클래스 인덱스, 신뢰도, 이미지 인덱스)]
                                            # groundtruths = [모든 이미지의 gt 박스의 수][3(좌표, 클래스 인덱스, 이미지 인덱스)]
    global max_gt_idx
    result = []

    for class_idx in range(21):                                 # 클래스 수만큼 반복
        preds = [d for d in predictions if d[1] == class_idx]   # d = [[xmin, ymin, xmax, ymax], class_idx, conf_score, img_idx]
        gts = [g for g in groundtruths if g[1] == class_idx]    # g = [[xmin, ymin, xmax, ymax], class_idx, img_idx]

        npos = len(gts) + 1e-10                                         # npos = 모든 이미지의 클래스 별 gt 박스 개수의 총합
        preds = sorted(preds, key = lambda conf : conf[2], reverse=True) # 모든 이미지의 클래스 별 예측 박스들을 conf_score에 따라 내림차순 정렬

        tp = np.zeros(len(preds))                               # 클래스별 모든 이미지의 총 예측 박스 수 만큼 0으로 채운 array
        fp = np.zeros(len(preds))

        det = Counter(cc[2] for cc in gts)  # 클래스 별 각 이미지의 gt수   {0: 1, 1: 2, ,,,}

        for key, val in det.items():        # 클래스별 각 이미지의 gt수 만큼 0인 배열  {0: [0], 1: [0, 0],...}
            det[key] = np.zeros(val)

        for pred_idx in range(len(preds)):                          # 클래스 별 모든 이미지의 예측 박스 수만큼 반복
            iou_max= 0
            gt = [gt for gt in gts if gt[2] == preds[pred_idx][3]]  # gt =  for문의 예측 박스에 해당하는 예측 박스와 같은 이미지의 gt 박스들
                                                                    # gt[2] = img_idx, preds[pred_idx][3] = img_idx
            for gt_idx in range(len(gt)):                           # gt 수만큼 반복
                iou = compute_iou(preds[pred_idx][0], gt[gt_idx][0])    # 예측 박스와 gt박스의 iou계산
                if iou > iou_max:                               # iou가 제일 높으면
                    iou_max = iou
                    max_gt_idx = gt_idx                         # gt_idx 저장

            if iou_max >= iou_threshold:                        # 가장 높은 iou가 임계점을 넘으면
                if det[preds[pred_idx][3]][max_gt_idx] == 0:    # 특정 gt박스가 매칭이 안됐으면
                    tp[pred_idx] = 1                            # 해당 예측 박스는 tp
                    det[preds[pred_idx][3]][max_gt_idx] = 1     # 매칭 체크
                else:                                           # 매칭이 안됐으면
                    fp[pred_idx] = 1                            # 해당 예측 박스는 fp
            else:                                               # iou가 임계점을 안넘으면
                fp[pred_idx] = 1                                # 해당 예측 박스는 fp

                                        # 모든 예측 박스에 대해 tp,fp를 채움
                                        # tp = [모든 예측 박스의 수] = [0,1,0,1,0,0,...,]
                                        # fp = [모든 예측 박스의 수] = [1,0,1,0,1,1,...,]
                                        # 둘은 0과 1이 상반됨
        cum_fp = np.cumsum(fp)          # cum_fp는 누적 합 = [모든 예측 박스의 수] = [0,1,1,2,2,2,3,3,4,...]
        cum_tp = np.cumsum(tp)          # cum_tp는 누적 합 = [모든 예측 박스의 수] = [0,1,1,2,2,2,3,3,4,...]

        recall = cum_tp / npos         # recall = [모든 예측 박스의 수] = 각 원소가 해당 순서의 누적 합을 전체 예측 박스로 나눈수 = [0/5000, 1/5000, 1/5000, 2/5000, ...]
        precision = np.divide(cum_tp, (cum_fp + cum_tp))    # precision = [모든 예측 박스의 수] = [0/(0+0), 1/(1+0), 1/(1+1),...,[3000/(3000+2000)]
        #print('recall', recall)
        #print('precision', precision)
        ap = ElevenPointInterpolatedAP(recall, precision)   # ap = [1] = 각 클래스별 ap
        result.append(ap)                                   # result = 각 클래스별 ap 리스트 = [0.8, 0.6, 0.7, ...] = [클래스 수]

    return result

def mean_average_precision(img_path_list, img_anno_list, iou_threshold = 0.5):

    prediction_list = []
    gt_list = []
    for i in range(400):                                                                 # mAP를 측정할 총 이미지 개수
        prediction_list.append(ssd_predict(img_path_list[i]))   # prediction_list = 한 이미지씩 경로를 읽어 모델 통과후 결과 값 모으기 [이미지 총 개수][한 이미지당 예측 박스의 수][3(좌표, 클래스 인덱스, 신뢰도)]
        img = cv2.imread(img_path_list[i])
        height, width, channels = img.shape                     # 이미지를 읽어 높이, 넓이 얻음
        gt_list.append(make_anno_list(img_anno_list[i], width, height, voc_classes))    # gt_list = gt정보 저장 [이미지 총 개수][한 이미지당 gt박스 수][2(좌표, 클래스 인덱스)]

    img_idx = 0
    prediction_idx_list = []
    gt_idx_list = []
    for gt, pred in zip(gt_list, prediction_list):  # gt = 한 이미지의 gt박스들 정보 [한 이미지당 gt박스 수][2(좌표, 클래스 인덱스)], pred = 한 이미지의 예측 박스들 정보 [한 이미지당 예측 박스의 수][3(좌표, 클래스 인덱스, 신뢰도)]
        for g in gt:                                # g = gt박스 하나의 정보[2(좌표, 클래스 인덱스)]
            g.append(img_idx)                       # g = g에 이미지 인덱스 추가[3(좌표, 클래스 인덱스, 이미지 인덱스)]
            gt_idx_list.append(g)                   # gt_idx_list = [모든 이미지의 gt 박스의 수][3(좌표, 클래스 인덱스, 이미지 인덱스)]

        for p in pred:                              # p = 예측 박스 하나의 정보[3(좌표, 클래스 인덱스, 신뢰도)]
            p.append(img_idx)                       # p = p에 이미지 인덱스 추가[4(좌표, 클래스 인덱스, 신뢰도, 이미지 인덱스)]
            prediction_idx_list.append(p)           # prediction_idx_list = [모든 이미지의 예측 박스의 수][4(좌표, 클래스 인덱스, 신뢰도, 이미지 인덱스)]
        img_idx += 1                    # 다음 이미지

    ap = AP(prediction_idx_list, gt_idx_list)       # ap = list[21] 각 클래스 별 ap값

    return ap
#print('train_img_list', train_img_list)
class_ap = mean_average_precision(train_img_list, train_anno_list)
ap = 0
print(class_ap)
for r in class_ap:
    ap += r
map = ap / len(class_ap)
print(map)
#####################################################################################################
#############웹캠 보이기

class SSDPredictShowCam():
    def __init__(self, eval_categories, net):
        self.eval_categories = eval_categories  # 클래스 이름 리스트
        self.net = net  # SSD 추론모드
        color_mean = (128, 128, 128)
        input_size = 300
        self.transform = DataTransform(input_size, color_mean)  # 데이터 전처리

    def show(self, image, data_confidence_level):
        predict_bbox, pre_dict_label_index, scores = self.ssd_predict(image,
                                                                      data_confidence_level)  # SSD 모델로 예측한 좌표와 클래스 반환
        self.vis_bbox(image, bbox=predict_bbox, label_index=pre_dict_label_index, scores=scores,
                      label_names=self.eval_categories)  # 예측한 정보로 웹캠에 그리기

    def ssd_predict(self, image, data_confidence_level):

        height, width, channels = image.shape
        phase = "val"
        img_transformed, boxes, labels = self.transform(image, phase, "", "")  # 이미지 전처리
        img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)  # 텐서로 변환
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        img = img.to(device)
        self.net.eval()
        x = img.unsqueeze(0)  # 배치 차원 추가：torch.Size([1, 3, 300, 300])

        detections = self.net(x)  # detections torch.Size([1, 21, 200, 5]) 각 클래스별 nms거친 스코어 상위 200개의 예측 박스 좌표와 스코어
        predict_bbox = []  # 예측한 바운딩 박스 좌표
        pre_dict_label_index = []  # 예측한 클래스 인덱스
        scores = []  # 예측한 스코어
        detections = detections.cpu().detach().numpy()  # autograd 끄고 넘파이로 변환

        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)  # 스코어가 data_confidence_level 이상인 박스 인덱스
        detections = detections[find_index]  # 스코어가 신뢰도 이상인 예측 박스 좌표와 스코어
        for i in range(len(find_index[1])):  # 예측한 물체의 개수만큼 반복
            if (find_index[1][i]) > 0:  # 예측한게 배경이 아니면
                sc = detections[i][0]
                bbox = detections[i][1:] * [width, height, width, height]
                label_ind = find_index[1][i] - 1

                predict_bbox.append(bbox)
                pre_dict_label_index.append(label_ind)
                scores.append(sc)
        return predict_bbox, pre_dict_label_index, scores

    def vis_bbox(self, img1, bbox, label_index, scores, label_names):
        num_classes = len(label_names)
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

        for i, bb in enumerate(bbox):
            if label_index[i] == 20 or label_index[i] == 14:
                label_name = label_names[label_index[i]]
                color = colors[label_index[i]]

                if scores is not None:
                    sc = scores[i]
                    display_txt1 = '%s: %.2f' % (label_name, sc)
                else:
                    display_txt1 = '%s: ans' % (label_name)

                xy = (int(bb[0]), int(bb[1]))
                width = int(bb[2] - bb[0])
                height = int(bb[3] - bb[1])
                pt1 = xy  # 바운딩 박스 시작 좌표
                pt2 = (xy[0] + width, xy[1] + height)  # 바운딩 박스 종료 좌표
                cv2.rectangle(img1, pt1, pt2, color, 2)  # 바운딩 박스 그리기
                cv2.putText(img1, display_txt1, (xy[0], xy[1] - 10), 0, 1, color, 1, cv2.LINE_AA)  # 예측 클래스 기재

#####################################################################################################

cap = cv2.VideoCapture(0)                                           # 웹캠에서 비디오 캡처

net.eval()                                                          # 네트워크를 추론 모드로
while True:
    ret, img_bgr = cap.read()                                       #  img_bgr = (640, 480)
    if not ret:
        break

    ssd = SSDPredictShowCam(eval_categories=voc_classes, net=net)   # 예측하고 
    ssd.show(img_bgr, data_confidence_level=0.7)                    # 그리기

    cv2.imshow('Object Detection', img_bgr)                 # 결과 화면 표시

    if cv2.waitKey(1) == ord('q'):                                  # 'q' 키를 누르면 종료
        break   

cap.release()                   # 자원 해제
cv2.destroyAllWindows()

#####################################################################################################
# 240913_1909 수정
######################