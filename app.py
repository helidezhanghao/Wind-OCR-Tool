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

st.set_page_config(page_title="力力的坐标工具 v17.0", page_icon="📍", layout="centered")

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
    """
    最基础的预处理：只做灰度化和二值化，不做任何去线骚操作
    """
    # 确保转为RGB
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
        
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 简单的二值化：小于阈值变黑，大于变白
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
    # 简单的清洗，不乱删东西
    text = text.replace('|', ' ').replace('[', ' ').replace(']', ' ')
    text = text.replace('°', ' ').replace("'", ' ').replace('"', ' ').replace(':', ' ')
    
    # 提取所有数字
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
    if 10000000 < x < 100000000 and y < 10000000: x, y = y, x
    y_str = str(int(y))
    final_cm = int(y_str[:2]) * 3 if len(y_str) == 8 else (cm if cm != 0 else 0)
    if final_cm == 0: return None, "Err"
    
    false_easting = 500000 + (int(y_str[:2]) * 1000000 if len(y_str) == 8 else 0)
    crs_str = f"+proj=tmerc +lat_0=0 +lon_0={final_cm} +k=1 +x_0={false_easting} +y_0=0 +ellps=GRS80 +units=m +no_defs"
    try:
        t = Transformer.from_crs(CRS.from_string(crs_str), CRS.from_epsg(4326), always_xy=True)
        lon, lat = t.transform(y, x)
        return lat, lon
    except: return None, "Error"

# ================= 界面主逻辑 =================

st.title("📍 力力的坐标工具 v17.0")

# --- 步骤 1: 上传 ---
st.header("1. 上传图片")
img_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

if img_file:
    # 只要上传新文件，强制重置状态
    if 'last_file' not in st.session_state or st.session_state.last_file != img_file.name:
        st.session_state.last_file = img_file.name
        st.session_state.raw_img = Image.open(img_file)
        st.session_state.angle = 0.0
        st.session_state.final_img = st.session_state.raw_img # 默认最终图就是原图
        st.rerun()

# 只有上传了图片才显示后续
if st.session_state.raw_img:
    
    st.divider()
    st.header("2. 图像处理 (可选)")
    
    # 裁切开关
    enable_crop = st.checkbox("✂️ 需要旋转或裁切？", value=False)
    
    if enable_crop:
        # ---- 裁切/旋转模式 ----
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("↺ 左旋90°"):
                st.session_state.angle += 90
                st.rerun()
        with c2:
            if st.button("↻ 右旋90°"):
                st.session_state.angle -= 90
                st.rerun()
        with c3:
            # 微调
            input_angle = st.number_input("微调角度", value=float(st.session_state.angle), step=0.5)
            if input_angle != st.session_state.angle:
                st.session_state.angle = input_angle
                st.rerun()

        # 实时显示旋转后的图供裁切
        rotated = rotate_image(st.session_state.raw_img, st.session_state.angle)
        st.caption("👇 拖动红框选中数据区域：")
        
        # 实时更新 cropped_img
        cropped_out = st_cropper(rotated, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
        
        # 实时把裁切结果给到 final_img
        st.session_state.final_img = cropped_out
        
        st.divider()
    else:
        # ---- 全图模式 ----
        # 不裁切时，直接重置 final_img 为原图 (如果之前裁切过，这里会恢复)
        st.session_state.final_img = st.session_state.raw_img
    
    
    # --- 步骤 3: 识别设置 (永远显示，不需要点按钮才出来) ---
    st.header("3. 识别参数")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚙️ 设置")
        # 坐标格式
        coord_mode = st.selectbox("坐标格式", 
                                  ["Decimal", "DMS", "DDM", "CGCS2000"],
                                  format_func=lambda x: {
                                      "Decimal": "🔢 纯小数 (82.78)",
                                      "DMS": "🌐 度分秒 (41°15'30\")",
                                      "DDM": "⏱️ 度+分 (41°15.5')",
                                      "CGCS2000": "📐 大地2000"
                                  }[x])
        
        # 大地坐标参数
        cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93,
