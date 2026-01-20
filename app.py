import streamlit as st
import simplekml
import re
from pyproj import CRS, Transformer
import os
from PIL import Image
import pytesseract
import shutil
import pandas as pd
import numpy as np
import cv2
from streamlit_cropper import st_cropper

# --- 环境配置 ---
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    tess_path = shutil.which("tesseract")
    if tess_path: pytesseract.pytesseract.tesseract_cmd = tess_path

st.set_page_config(page_title="力力的坐标工具 v17.1", page_icon="📍", layout="centered")

# --- 状态初始化 ---
if 'angle' not in st.session_state:
    st.session_state.angle = 0.0
if 'raw_img' not in st.session_state:
    st.session_state.raw_img = None
if 'final_img' not in st.session_state:
    st.session_state.final_img = None
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = ""

# --- 核心工具函数 ---
def rotate_image(image, angle):
    """无损旋转"""
    return image.rotate(angle, expand=True)

def simple_preprocess(pil_image, threshold):
    """最基础的预处理"""
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary)

def smart_fix_decimal(val):
    if val > 180 and val < 200000000: 
        s_val = str(int(val))
        if len(s_val) >= 4:
            v2 = float(s_val[:2] + "." + s_val[2:])
            if 3 < v2 < 180: return v2
            v3 = float(s_val[:3] + "." + s_val[3:])
            if 3 < v3 < 180: return v3
    return val

def dms_to_dec(d, m, s):
    return float(d) + float(m)/60 + float(s)/3600

def ddm_to_dec(d, m):
    return float(d) + float(m)/60

def extract_coords(text, mode):
    text = text.replace('|', ' ').replace('[', ' ').replace(']', ' ')
    text = text.replace('°', ' ').replace("'", ' ').replace('"', ' ').replace(':', ' ')
    raw_nums = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", text)
    nums_val = [float(n) for n in raw_nums]
    data = []

    if mode == "Decimal":
        fixed_nums = [smart_fix_decimal(n) for n in nums_val]
        valid_indices = [i for i, n in enumerate(fixed_nums) if 3 < abs(n) < 180]
        for i in range(0, len(valid_indices) - 1, 2):
            idx1, idx2 = valid_indices[i], valid_indices[i+1]
            data.append({"纬度/X": fixed_nums[idx1], "经度/Y": fixed_nums[idx2]})
            
    elif mode == "DMS": 
        if len(nums_val) >= 6:
            for i in range(len(nums_val) - 5):
                g = nums_val[i:i+6]
                if (abs(g[0])<180 and g[1]<60 and g[2]<60 and 
                    abs(g[3])<180 and g[4]<60 and g[5]<60):
                    lat = dms_to_dec(g[0], g[1], g[2])
                    lon = dms_to_dec(g[3], g[4], g[5])
                    data.append({"纬度/X": lat, "经度/Y": lon})

    elif mode == "DDM": 
        if len(nums_val) >= 4:
            for i in range(len(nums_val) - 3):
                g = nums_val[i:i+4]
                if (abs(g[0])<180 and g[1]<60 and abs(g[2])<180 and g[3]<60):
                    lat = ddm_to_dec(g[0], g[1])
                    lon = ddm_to_dec(g[2], g[3])
                    data.append({"纬度/X": lat, "经度/Y": lon})

    elif mode == "CGCS2000":
        valid_indices = [i for i, n in enumerate(nums_val) if abs(n) > 300000]
        for i in range(0, len(valid_indices) - 1, 2):
            idx1, idx2 = valid_indices[i], valid_indices[i+1]
            data.append({"纬度/X": nums_val[idx1], "经度/Y": nums_val[idx2]})
    return data

def to_wgs84(v1, v2, cm, swap):
    x, y = (v2, v1) if swap else (v1, v2)
    if 10000000 < x < 1000
