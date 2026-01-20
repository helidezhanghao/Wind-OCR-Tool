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
# 引入新的裁切组件库
from streamlit_cropper import st_cropper

# --- 环境配置 ---
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    tess_path = shutil.which("tesseract")
    if tess_path: pytesseract.pytesseract.tesseract_cmd = tess_path

st.set_page_config(page_title="风资源坐标神器v13.0", page_icon="✂️", layout="centered")

# --- 状态初始化 ---
if 'rotation_angle' not in st.session_state:
    st.session_state.rotation_angle = 0
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'final_edited_image' not in st.session_state:
    st.session_state.final_edited_image = None
if 'raw_ocr_text' not in st.session_state:
    st.session_state.raw_ocr_text = ""

# --- 核心算法 (保持 v12 不变) ---
def smart_fix_coordinate(val):
    if val > 180 and val < 200000000: 
        s_val = str(int(val))
        if len(s_val) >= 4:
            v2 = float(s_val[:2] + "." + s_val[2:])
            if 3 < v2 < 180: return v2
            v3 = float(s_val[:3] + "." + s_val[3:])
            if 3 < v3 < 180: return v3
    return val

def process_image_v13(pil_image, color_strategy, threshold, remove_line):
    # 确保输入是 RGB
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    gray = None
    b, g, r = cv2.split(img_cv)
    if color_strategy == "🤖 自动(智能去底色)":
        means = [np.mean(b), np.mean(g), np.mean(r)]
        max_idx = np.argmax(means)
        if max_idx == 0: gray = b
        elif max_idx == 1: gray = g
        else: gray = r
    elif color_strategy == "🔵 强制蓝底模式": gray = b
    elif color_strategy == "🔴 强制红/黄底模式": gray = r
    elif color_strategy == "🟢 强制绿底模式": gray = g
    else: gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    if remove_line:
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        lines_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h, iterations=1)
        lines_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v, iterations=1)
        binary[lines_h==255] = 255
        binary[lines_v==255] = 255

    return Image.fromarray(binary)

def extract_coords_smart(text, mode):
    text = text.replace('|', ' ').replace('[', ' ').replace(']', ' ')
    raw_nums = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", text)
    data = []
    nums_val = []
    for n in raw_nums:
        v = float(n)
        if mode == "Decimal": v = smart_fix_coordinate(v)
        nums_val.append(v)
    
    if mode == "Decimal":
        valid_indices = [i for i, n in enumerate(nums_val) if 3 < abs(n) < 180]
        for i in range(0, len(valid_indices) - 1, 2):
            idx1 = valid_indices[i]
            idx2 = valid_indices[i+1]
            data.append({"纬度/X": nums_val[idx1], "经度/Y": nums_val[idx2]})
            
    elif mode == "CGCS2000":
        valid_indices = [i for i, n in enumerate(nums_val) if abs(n) > 300000]
        for i in range(0, len(valid_indices) - 1, 2):
            data.append({"纬度/X": nums_val[valid_indices[i]], "经度/Y": nums_val[valid_indices[i+1]]})
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
st.title("✂️ 风资源坐标神器 v13.0")
st.caption("完美交互版：先旋转裁切，再精准识别")

# 侧边栏配置
with st.sidebar:
    st.header("1. 识别设置")
    coord_mode = st.selectbox("坐标格式", ["Decimal", "CGCS2000", "DMS", "DDM"])
    cm_val = 0
    force_swap = False
    if coord_mode == "CGCS2000":
        cm_options = {"自动": 0, "75": 75, "81": 81, "87": 87, "93": 93, "99": 99, "105": 105, "114": 114, "123": 123}
        cm_val = cm_options[st.selectbox("中央经线", list(cm_options.keys()))]
        force_swap = st.checkbox("强制交换XY")
        
    st.header("2. 图像预处理参数")
    bg_strategy = st.selectbox("底色处理策略", ["🤖 自动(智能去底色)", "🔵 强制蓝底模式", "🔴 强制红/黄底模式", "⚫ 普通黑白模式"])
    thresh = st.slider("黑白阈值", 0, 255, 140)
    remove_line = st.checkbox("尝试抹除表格线", value=True)

# --- 主流程 ---
img_file = st.file_uploader("📸 第一步：上传图片", type=['png', 'jpg', 'jpeg'])

# 如果有新图片上传，重置状态
if img_file and (st.session_state.uploaded_image is None or img_file.name != st.session_state.uploaded_image_name):
    st.session_state.uploaded_image = Image.open(img_file)
    st.session_state.uploaded_image_name = img_file.name
    st.session_state.rotation_angle = 0
    st.session_state.final_edited_image = None
    st.session_state.raw_ocr_text = ""
    st.rerun()

# --- 编辑模式 ---
if st.session_state.uploaded_image and st.session_state.final_edited_image is None:
    st.divider()
    st.subheader("✂️ 第二步：旋转与裁切 (重要!)")
    st.info("请先把图片转正，然后拖动方框，只保留最核心的表格数据区域。")
    
    # 旋转控制栏
    col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
    with col_r1: 
        if st.button("⬅️ 左旋90°"): st.session_state.rotation_angle += 90
    with col_r2:
        if st.button("🔄 左微调"): st.session_state.rotation_angle += 1
    with col_r3:
        st.write(f"当前角度: {st.session_state.rotation_angle % 360}°")
    with col_r4:
        if st.button("🔄 右微调"): st.session_state.rotation_angle -= 1
    with col_r5:
        if st.button("➡️ 右旋90°"): st.session_state.rotation_angle -= 90
        
    # 应用旋转
    rotated_img = st.session_state.uploaded_image.rotate(st.session_state.rotation_angle, expand=True)
    
    # 裁切组件
    # realtime_update=True 让选框拖动时实时显示结果
    cropped_img = st_cropper(rotated_img, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    
    st.write("预览裁切结果：")
    st.image(cropped_img, caption="最终用于识别的图像区域", width=300)
    
    if st.button("✅ 确认裁切并继续", type="primary", use_container_width=True):
        st.session_state.final_edited_image = cropped_img
        st.rerun()

# --- 识别与生成模式 ---
if st.session_state.final_edited_image:
    st.divider()
    st.subheader("🧐 第三步：预处理与提取")
    
    # 对裁切后的图进行预处理
    processed_img = process_image_v13(st.session_state.final_edited_image, bg_strategy, thresh, remove_line)
    
    c1, c2 = st.columns(2)
    c1.image(st.session_state.final_edited_image, caption="裁切后原图")
    c2.image(processed_img, caption="机器看到的 (去底色/去线后)")
    
    if st.button("🔥 开始OCR提取文字", type="primary", use_container_width=True):
        with st.spinner("正在精准识别..."):
            # 使用 PSM 6 适合表格块
            text = pytesseract.image_to_string(processed_img, lang='eng', config='--psm 6')
            st.session_state.raw_ocr_text = text
            st.rerun()

# 文本修正与生成 (与v12相同)
if st.session_state.raw_ocr_text:
    st.divider()
    st.subheader("📝 第四步：结果确认与下载")
    
    raw_data = extract_coords_smart(st.session_state.raw_ocr_text, coord_mode)
    
    if raw_data:
        df = pd.DataFrame(raw_data)
        st.session_state.df = df
        st.success(f"成功提取 {len(raw_data)} 行数据！")
        
        edited_df = st.data_editor(st.session_state.df, num_rows="dynamic", use_container_width=True)
        
        if st.button("🚀 生成 KMZ"):
            kml = simplekml.Kml()
            cnt = 0
            for idx, row in edited_df.iterrows():
                try:
                    v1 = float(row["纬度/X"])
                    v2 = float(row["经度/Y"])
                    lat, lon = 0, 0
                    if coord_mode == "Decimal":
                        lat, lon = v1, v2
                        if lat > lon and lat < 180: lat, lon = lon, lat 
                        if lat > 60: lat, lon = lon, lat 
                    elif coord_mode == "CGCS2000":
                        res, msg = cgcs2000_to_wgs84(v1, v2, cm_val, force_swap)
                        if res: lat, lon = res, msg
                        else: continue
                    else:
                        lat, lon = v1, v2
                    kml.newpoint(name=f"P{idx+1}", coords=[(lon, lat)])
                    cnt += 1
                except: continue
            
            if cnt > 0:
                kml.save("points.kmz")
                with open("points.kmz", "rb") as f:
                    st.download_button("下载 KMZ", f, "Coordinates.kmz", type="primary")
            else:
                st.warning("无有效数据")
    else:
        st.error("未找到有效数据，请检查识别结果 👇")
        st.text_area("OCR 原始内容", st.session_state.raw_ocr_text)

# 添加一个重置按钮，方便重新上传
if st.session_state.final_edited_image:
    st.divider()
    if st.button("🔄 重新上传/编辑图片"):
        st.session_state.uploaded_image = None
        st.session_state.final_edited_image = None
        st.session_state.raw_ocr_text = ""
        st.rerun()
