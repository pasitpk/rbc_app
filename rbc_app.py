import os
import json
import numpy as np
import pandas as pd
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
IDX_NAME = configs['IDX_NAME']
TARGET_NAME = configs['TARGET_NAME']
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
    cv2.rectangle(img, (int(x0), int(y0)), 
                        (int(x0+w), int(y0+h)), color, thickness)
    return img


def draw_bboxes(img, bboxes):
    img = img.copy()
    progress_bar = st.progress(0)
    for i, bbox in enumerate(bboxes):
        draw_bbox(img, bbox)
        progress_bar.progress(int((i+1)/len(bboxes) * 100))
    progress_bar.empty()
    return img    


def bboxes_to_csv(bboxes, img_name):
    df = pd.DataFrame(bboxes)
    df['x0'] = df['bbox'].apply(lambda x: x[0])
    df['y0'] = df['bbox'].apply(lambda x: x[1])
    df['w'] = df['bbox'].apply(lambda x: x[2])
    df['h'] = df['bbox'].apply(lambda x: x[3])
    df['img_name'] = img_name
    return df[['img_name', 'category_id', 'score', 'x0', 'y0', 'w', 'h']]
   

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    provider = PROVIDER if PROVIDER in ort.get_available_providers() else 'CPUExecutionProvider'
    print('using provider "{}"'.format(provider))
    model = ort.InferenceSession(model_path, providers=[provider])
    return model


@st.cache(allow_output_mutation=True, suppress_st_warning=True, show_spinner=False)
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

    with st.spinner(text="Processing ..."):
        progress_bar = st.progress(0)
        for b in range(total_batch):
            outs = model.run(None, {model.get_inputs()[0].name: x[b*bs:(b+1)*bs]})
            out_bboxes.append(outs[0])
            out_cls.append(outs[1])
            progress_bar.progress(int((b+1)/total_batch * 100))
    
    progress_bar.empty()

    if DYNAMIC_EXPORT:
        outs = [np.concatenate(out_bboxes), np.concatenate(out_cls)]
    else:
        outs = [np.stack(out_bboxes), np.stack(out_cls)]

    with st.spinner(text="Post-processing ..."):
        bboxes = onnx_to_bbox(merge_bbox(idx, outs, scales), IOU_THRESHOLD, SCORE_THRESHOLD)

    with st.spinner(text="Drawing bboxes ..."):
        bbox_img = draw_bboxes(img[..., ::-1], bboxes)     

    return bbox_img, bboxes


def show_statistics(bbox_df):

    st.text(f'Total number of red blood cells = {len(bbox_df):,}')

    counts = bbox_df['category_id'].value_counts()
    counts.index = counts.index.astype('str')
    name_count = dict()
    
    for idx, name in IDX_NAME.items():
        name_count[name] = counts.loc[idx] if idx in counts.index else 0  
        st.text(f'  {name} : {name_count[name]:,}')
    
    st.text(f'{int(name_count[TARGET_NAME] / len(bbox_df) * 1000)} {TARGET_NAME} : 1,000 red blood cells')
    

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
    
    image_file = st.file_uploader(f"Minimum image size is {WINDOW_SIZE[0]} x {WINDOW_SIZE[1]} pixels", type=['jpg', 'png', 'jpeg'])

    if image_file is not None:        

        image = np.asarray(Image.open(image_file))

        if image.shape[0] < WINDOW_SIZE[0] or image.shape[1] < WINDOW_SIZE[1]:

            st.error(f'The image resolution must be at least {WINDOW_SIZE[0]} x {WINDOW_SIZE[1]} pixels, but got {image.shape[0]} x {image.shape[1]} pixels')

            if st.button('Clear'):
                st.markdown('<meta http-equiv="refresh" content="0.1" >', unsafe_allow_html=True)

        else:

            result_img, bboxes= detect_rbc(model, image)
            bbox_df = bboxes_to_csv(bboxes, image_file.name)

            st.text("Original Image")
            st.image(image)

            st.text("Prediction")
            st.image(result_img)

            show_statistics(bbox_df)

            if st.button('Clear'):
                st.markdown('<meta http-equiv="refresh" content="0.1" >', unsafe_allow_html=True)


if __name__ == '__main__':
    model = load_model()
    main()
