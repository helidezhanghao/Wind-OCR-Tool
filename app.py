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

# --- 环境配置 ---
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    tess_path = shutil.which("tesseract")
    if tess_path: pytesseract.pytesseract.tesseract_cmd = tess_path

st.set_page_config(page_title="风资源坐标神器v10.0", page_icon="💀", layout="centered")

# --- 核心算法 ---
def process_image_v10(pil_image, mode_type, threshold, remove_line):
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 1. 二值化策略
    if mode_type == "自动(适应蓝底/阴影)":
        # 自适应阈值，专治光线不均
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 31, 15)
    else:
        # 手动阈值，专治字迹太淡
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 2. 暴力去线 (可选，如果字被线切断了就关掉它)
    if remove_line:
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        lines_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h, iterations=1)
        lines_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v, iterations=1)
        # 变白
        binary[lines_h==255] = 255
        binary[lines_v==255] = 255

    return Image.fromarray(binary)

def extract_coords_from_text(text, mode):
    # 清洗
    text = text.replace('°', ' ').replace("'", ' ').replace('"', ' ').replace(':', ' ')
    text = text.replace('l', '1').replace('O', '0').replace('o', '0')
    text = text.replace('|', ' ').replace('[', ' ').replace(']', ' ')
    
    # 提取所有数字 (保留原样字符串，防止精度丢失)
    # 逻辑：匹配像浮点数的东西
    raw_nums = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", text)
    
    data = []
    
    # 转为 float 进行逻辑判断，但存储 string
    nums_val = [float(n) for n in raw_nums]
    
    if mode == "Decimal":
        # 找 3 < x < 180 的数
        valid_indices = [i for i, n in enumerate(nums_val) if 3 < abs(n) < 180]
        # 两两配对
        for i in range(0, len(valid_indices) - 1, 2):
            idx1 = valid_indices[i]
            idx2 = valid_indices[i+1]
            # 存储为字符串，保证小数点不丢失
            data.append({"纬度/X": raw_nums[idx1], "经度/Y": raw_nums[idx2]})
            
    elif mode == "CGCS2000":
        valid_indices = [i for i, n in enumerate(nums_val) if abs(n) > 300000]
        for i in range(0, len(valid_indices) - 1, 2):
            idx1 = valid_indices[i]
            idx2 = valid_indices[i+1]
            data.append({"纬度/X": raw_nums[idx1], "经度/Y": raw_nums[idx2]})
            
    # DMS/DDM 比较复杂，暂只支持 Decimal 和 2000 的暴力提取
    return data

def cgcs2000_to_wgs84(v1, v2, cm_val, force_swap):
    x, y = (v2, v1) if force_swap else (v1, v2)
    if 10000000 < x < 100000000 and y < 10000000: x, y = y, x 
    y_str = str(int(y))
    final_cm = 0
    if len(y_str) == 8: final_cm = int(y_str[:2]) * 3
    elif cm_val != 0: final_cm = cm_val
    else: return None, "需选区域"
    false_easting = 500000 + (int(y_str[:2]) * 1000000 if len(y_str) == 8 else 0)
    crs_str = f"+proj=tmerc +lat_0=0 +lon_0={final_cm} +k=1 +x_0={false_easting} +y_0=0 +ellps=GRS80 +units=m +no_defs"
    try:
        transformer = Transformer.from_crs(CRS.from_string(crs_str), CRS.from_epsg(4326), always_xy=True)
        lon, lat = transformer.transform(y, x)
        return lat, lon
    except: return None, "转换错"

# --- 界面 ---
st.title("💀 风资源坐标神器 v10.0")
st.caption("人机合一模式：OCR识别 -> 人工修正 -> 生成")

with st.sidebar:
    st.header("1. 图像处理")
    proc_mode = st.selectbox("处理模式", ["自动(适应蓝底/阴影)", "手动(调节黑白阈值)"])
    thresh = 127
    if proc_mode == "手动(调节黑白阈值)":
        thresh = st.slider("黑白阈值", 0, 255, 120)
    
    remove_line = st.checkbox("尝试抹除表格线", value=False, help="如果数字被线切断了，请取消勾选此项")
    
    st.header("2. 坐标参数")
    coord_mode = st.selectbox("坐标格式", ["Decimal", "CGCS2000", "DMS", "DDM"])
    
    cm_val = 0
    force_swap = False
    if coord_mode == "CGCS2000":
        cm_options = {"自动": 0, "75": 75, "81": 81, "87": 87, "93": 93, "99": 99, "105": 105, "114": 114, "123": 123}
        cm_val = cm_options[st.selectbox("中央经线", list(cm_options.keys()))]
        force_swap = st.checkbox("强制交换XY")

img_file = st.file_uploader("📸 上传图片", type=['png', 'jpg', 'jpeg'])

if 'raw_ocr_text' not in st.session_state:
    st.session_state.raw_ocr_text = ""

if img_file:
    image = Image.open(img_file)
    # 图像处理预览
    processed_img = process_image_v10(image, proc_mode, thresh, remove_line)
    st.image(processed_img, caption="机器眼中的图 (如果不清晰，请调整左侧设置)", use_column_width=True)
    
    if st.button("🔥 第一步：提取文字", type="primary"):
        with st.spinner("OCR 扫描中..."):
            # 识别
            text = pytesseract.image_to_string(processed_img, lang='eng', config='--psm 6')
            st.session_state.raw_ocr_text = text
            st.rerun()

# 文本修正区 (核心功能)
if st.session_state.raw_ocr_text:
    st.divider()
    st.subheader("📝 第二步：修正识别结果")
    st.caption("如果在下面看到乱码，请直接在这里修改！比如把 l 改成 1，补上小数点。")
    
    # 让用户可以编辑 OCR 的原始文本
    user_edited_text = st.text_area("OCR 原始文本 (可编辑)", 
                                  value=st.session_state.raw_ocr_text, 
                                  height=200)
    
    if st.button("⚡ 第三步：解析并生成表格"):
        raw_data = extract_coords_from_text(user_edited_text, coord_mode)
        
        if raw_data:
            df = pd.DataFrame(raw_data)
            st.session_state.df = df
            st.success(f"成功提取 {len(raw_data)} 组坐标！")
        else:
            st.error("未在文本中提取到有效坐标，请检查上面的文本是否包含数字。")

# 结果生成区
if 'df' in st.session_state and not st.session_state.df.empty:
    st.divider()
    st.subheader("🚀 第四步：下载 KMZ")
    
    # 强制显示为字符串，防止显示时精度丢失
    st.data_editor(st.session_state.df, num_rows="dynamic")
    
    if st.button("📥 生成最终文件"):
        kml = simplekml.Kml()
        cnt = 0
        for idx, row in st.session_state.df.iterrows():
            try:
                # 转 float 计算
                v1 = float(row["纬度/X"])
                v2 = float(row["经度/Y"])
                lat, lon = 0, 0
                
                # 简单归位
                if coord_mode == "Decimal":
                    lat, lon = v1, v2
                    if lat > lon and lat < 180: lat, lon = lon, lat # 中国区经度通常大
                    if lat > 60: lat, lon = lon, lat # 再次保险
                elif coord_mode == "CGCS2000":
                    res, msg = cgcs2000_to_wgs84(v1, v2, cm_val, force_swap)
                    if res: lat, lon = res, msg
                    else: continue
                else:
                    # DMS/DDM 在文本编辑阶段建议直接手动改为小数，或者这里简单处理
                    lat, lon = v1, v2
                
                kml.newpoint(name=f"P{idx+1}", coords=[(lon, lat)])
                cnt += 1
            except: continue
        
        if cnt > 0:
            kml.save("points.kmz")
            with open("points.kmz", "rb") as f:
                st.download_button("点击下载 KMZ", f, "坐标.kmz", type="primary")
