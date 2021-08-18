import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import mmcv
import onnxruntime as ort
from PIL import Image
import streamlit as st
import copy

ort.set_default_logger_severity(3)

with open('configs.json', 'r') as f:
    configs = json.load(f)

model_path = configs['MODEL']
PROVIDER = configs['PROVIDER']
DYNAMIC_EXPORT = configs['DYNAMIC_EXPORT']
BATCH_SIZE = configs['BATCH_SIZE']
WINDOW_SIZE = configs['WINDOW_SIZE']  # (h, w)
OVERLAP = configs['OVERLAP']  # (left-right, up-down)
SCALE = tuple(configs['SCALE'])
MEAN = np.asarray(configs['MEAN'])
SD = np.asarray(configs['SD'])
TO_RGB = configs['TO_RGB']
SIZE_DIVISOR = configs['SIZE_DIVISOR']
LENGTH_LIMIT = configs['LENGTH_LIMIT']
IOU_THRESHOLD = configs['IOU_THRESHOLD']
SCORE_THRESHOLD = configs['SCORE_THRESHOLD']
CLASS_W = configs['CLASS_W']
EDGE_COLOR = configs['EDGE_COLOR']
EDGE_THICKNESS = configs['EDGE_THICKNESS']



def window_image(img, window_size, overlap):
    img_h, img_w, _ = img.shape
    n_rows = 1 - (-(img_h - window_size[0]) // (window_size[0] - overlap[1]))
    n_cols = 1 - (-(img_w - window_size[1]) // (window_size[1] - overlap[0]))
    
    sub_idx = []
    sub_imgs = []
    
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * (window_size[0] - overlap[1]) if r + 1 < n_rows else img_h - window_size[0]
            j = c * (window_size[1] - overlap[0]) if c + 1 < n_cols else img_w - window_size[1]
            sub_idx.append([i, j])
            sub_imgs.append(img[i:i+window_size[0], j:j+window_size[1]])

    return sub_idx, sub_imgs


def transform(img, target_scale):
    img, scale = mmcv.imrescale(img, target_scale, return_scale=True)
    return mmcv.impad_to_multiple(mmcv.imnormalize(img, MEAN, SD, TO_RGB), SIZE_DIVISOR), scale


def preprocess(img, window_size, overlap, target_scale=SCALE):
    sub_idx, sub_imgs = window_image(img, window_size, overlap)
    transformed_imgs = []
    scales = []
    for sub_img in sub_imgs:
        trans_img, scale = transform(sub_img, target_scale)
        transformed_imgs.append(trans_img)
        scales.append(scale)
    transformed_imgs = np.stack(transformed_imgs)
    return sub_idx, np.moveaxis(transformed_imgs, -1, 1), scales


def re_weight(results, cls_w):
    new_results = copy.deepcopy(results)
    for i, w in enumerate(cls_w):
            cls_idx = new_results[1] == i
            new_results[0][:, :, -1][cls_idx] = np.clip(w * new_results[0][:, :, -1][cls_idx], 0, 1)
    return new_results


def merge_bbox(sub_idx, results, scales):
    results = copy.deepcopy(results)
    bboxes = []
    cls = []
    for (i, j), res_bbox, res_cls, scale in zip(sub_idx, results[0], results[1], scales):
        cls.append(res_cls)
        for bbox in res_bbox:
            bbox[0] = (bbox[0] / scale) + j
            bbox[1] = (bbox[1] / scale) + i
            bbox[2] = (bbox[2] / scale) + j
            bbox[3] = (bbox[3] / scale) + i
        bboxes.append(res_bbox)
    bboxes = np.concatenate(bboxes)
    cls = np.concatenate(cls)
    return [bboxes, cls]


def ltrb_to_xywh(ltrb_array):
    xywh_array = ltrb_array.copy()
    xywh_array[:, 2:4] = ltrb_array[:, 2:4] - ltrb_array[:, :2]
    return xywh_array


def filter_bboxes(bboxes):
    """
    for bbox in "xywh" format
    """
    indices = np.argwhere(
      (LENGTH_LIMIT[0] < bboxes[:, 2]) & (bboxes[:, 2] < LENGTH_LIMIT[1]) 
    & (LENGTH_LIMIT[0] < bboxes[:, 3]) & (bboxes[:, 3] < LENGTH_LIMIT[1])
    )
    return indices.flatten()


def onnx_to_bbox(onnx_res, iou_thresh=0.5, score_thresh=0.05):
    bboxes = ltrb_to_xywh(onnx_res[0])
    cls = onnx_res[1]
    indices = filter_bboxes(bboxes)
    bboxes, cls = bboxes[indices], cls[indices]
    indices = cv2.dnn.NMSBoxes(bboxes[:, :-1].tolist(), bboxes[:, -1].tolist(), score_thresh, iou_thresh).flatten()
    bboxes = [{'bbox': [bb[0], bb[1], bb[2], bb[3]], 'score': bb[4], 'category_id': c+1} for i, (bb, c) in enumerate(zip(bboxes, cls)) if i in indices]
    return bboxes    


def draw_bbox(img, bbox):
    color = EDGE_COLOR[str(bbox['category_id'])]
    thickness = EDGE_THICKNESS
    x0, y0, w, h = bbox['bbox']
    img = cv2.rectangle(img.copy(), (int(x0), int(y0)), 
                        (int(x0+w), int(y0+h)), color, thickness)
    return img


def draw_bboxes(img, bboxes):
    for bbox in bboxes:
        img = draw_bbox(img, bbox)
    return img    


def bboxes_to_csv(bboxes, img_name):
    df = pd.DataFrame(bboxes)
    df['x0'] = df['bbox'].apply(lambda x: x[0])
    df['y0'] = df['bbox'].apply(lambda x: x[1])
    df['w'] = df['bbox'].apply(lambda x: x[2])
    df['h'] = df['bbox'].apply(lambda x: x[3])
    df['img_name'] = img_name
    return df[['img_name', 'category_id', 'score', 'x0', 'y0', 'w', 'h']]


def detect_rbc(model, img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    idx, x, scales = preprocess(img, WINDOW_SIZE, OVERLAP)

    if DYNAMIC_EXPORT:
        bs = BATCH_SIZE
    else:
        bs = 1    

    out_bboxes = []
    out_cls = []
    total_batch = -(-len(x) // bs)

    for b in range(total_batch):
        outs = model.run(None, {model.get_inputs()[0].name: x[b*bs:(b+1)*bs]})
        out_bboxes.append(outs[0])
        out_cls.append(outs[1])
    
    if DYNAMIC_EXPORT:
        outs = [np.concatenate(out_bboxes), np.concatenate(out_cls)]
    else:
        outs = [np.stack(out_bboxes), np.stack(out_cls)]

    bboxes = onnx_to_bbox(merge_bbox(idx, outs, scales), IOU_THRESHOLD, SCORE_THRESHOLD)

    bbox_img = draw_bboxes(img[..., ::-1], bboxes)

    return bbox_img


def main():

    st.title("AI4PBS")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:#EC7063 ;padding:10px">
    <h2 style="color:white;text-align:center;">Schistocyte Detection</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        image = np.asarray(Image.open(image_file))
        st.text("Original Image")
        st.image(image)
        result_img= detect_rbc(model, image)
        st.text("Prediction")
        st.image(result_img)

if __name__ == '__main__':
    provider = PROVIDER if PROVIDER in ort.get_available_providers() else 'CPUExecutionProvider'
    print('using provider "{}"'.format(provider))
    model = ort.InferenceSession(model_path, providers=[provider])
    main()
