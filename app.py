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

st.set_page_config(page_title="风资源坐标神器v14.0", page_icon="💀", layout="centered")

# --- 状态初始化 ---
if 'angle' not in st.session_state:
    st.session_state.angle = 0.0
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'cropped_img' not in st.session_state:
    st.session_state.cropped_img = None
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = ""

# --- 核心工具函数 ---
def rotate_image(image, angle):
    """无损旋转"""
    return image.rotate(angle, expand=True)

def visualize_lines(pil_image, line_thickness, threshold):
    """
    可视化去线：返回 (预览红线图, 最终去线图)
    """
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 提取线条掩膜
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(line_thickness * 10), 1))
    mask_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h, iterations=1)
    
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(line_thickness * 10)))
    mask_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v, iterations=1)
    
    # 合并掩膜
    mask_lines = cv2.bitwise_or(mask_h, mask_v)
    
    # 1. 预览图：把线涂红
    preview = img_cv.copy()
    preview[mask_lines == 255] = [0, 0, 255] # BGR Red
    
    # 2. 结果图：把线涂白
    clean_binary = binary.copy()
    clean_binary[mask_lines == 255] = 255
    
    return Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)), Image.fromarray(clean_binary)

def smart_fix_coordinate(val):
    """智能修复丢失的小数点"""
    if val > 180 and val < 200000000: 
        s_val = str(int(val))
        if len(s_val) >= 4:
            # 尝试在第2位后加点
            v2 = float(s_val[:2] + "." + s_val[2:])
            if 3 < v2 < 180: return v2
            # 尝试在第3位后加点
            v3 = float(s_val[:3] + "." + s_val[3:])
            if 3 < v3 < 180: return v3
    return val

def extract_coords(text, mode):
    # 清洗
    text = text.replace('|', ' ').replace('[', ' ').replace(']', ' ')
    raw_nums = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", text)
    data = []
    nums_val = []
    for n in raw_nums:
        v = float(n)
        if mode == "Decimal": v = smart_fix_coordinate(v)
        nums_val.append(v)
    
    if mode == "Decimal":
        # 找 3 < x < 180
        valid_indices = [i for i, n in enumerate(nums_val) if 3 < abs(n) < 180]
        for i in range(0, len(valid_indices) - 1, 2):
            data.append({"纬度/X": nums_val[valid_indices[i]], "经度/Y": nums_val[valid_indices[i+1]]})
    elif mode == "CGCS2000":
        # 找大数
        valid_indices = [i for i, n in enumerate(nums_val) if abs(n) > 300000]
        for i in range(0, len(valid_indices) - 1, 2):
            data.append({"纬度/X": nums_val[valid_indices[i]], "经度/Y": nums_val[valid_indices[i+1]]})
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

st.title("💀 风资源坐标神器 v14.0")

# --- 步骤 1: 上传 ---
st.header("1️⃣ 上传图片")
img_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

if img_file:
    # 只要上传新文件，强制重置
    if 'last_file' not in st.session_state or st.session_state.last_file != img_file.name:
        st.session_state.last_file = img_file.name
        st.session_state.raw_img = Image.open(img_file)
        st.session_state.angle = 0.0
        st.session_state.step = 2
        st.session_state.cropped_img = None
        st.rerun()

# --- 步骤 2: 旋转 & 裁切 ---
if st.session_state.step >= 2 and 'raw_img' in st.session_state:
    st.divider()
    st.header("2️⃣ 旋转 & 裁切")
    st.info("👇 先把图转正，然后拖动红框选中数据！")
    
    # 旋转控制 (解决互斥问题的关键：都操作同一个 state.angle)
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("↺ 左旋90°", use_container_width=True):
            st.session_state.angle += 90
            st.rerun()
    with c2:
        if st.button("↻ 右旋90°", use_container_width=True):
            st.session_state.angle -= 90
            st.rerun()
    with c3:
        # 数字输入框直接绑定 state.angle，实现任意微调
        input_angle = st.number_input("精确角度微调 (支持小数)", value=float(st.session_state.angle), step=0.5)
        if input_angle != st.session_state.angle:
            st.session_state.angle = input_angle
            st.rerun()

    # 执行旋转
    rotated = rotate_image(st.session_state.raw_img, st.session_state.angle)
    
    # 裁切控件
    cropped = st_cropper(rotated, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    
    if st.button("✂️ 确认裁切，下一步", type="primary", use_container_width=True):
        st.session_state.cropped_img = cropped
        st.session_state.step = 3
        st.rerun()

# --- 步骤 3: 可视化去线 ---
if st.session_state.step >= 3 and st.session_state.cropped_img:
    st.divider()
    st.header("3️⃣ 调整去表格线")
    st.caption("🔴 红色 = 即将删除的内容。请调整滑块，确保红色只覆盖线，不覆盖字！")
    
    col_ctrl, col_view = st.columns([1, 2])
    with col_ctrl:
        thresh = st.slider("黑白阈值", 0, 255, 140)
        line_w = st.slider("线条粗细 (红线宽度)", 1, 10, 4)
        
        # 格式选择放在这里
        st.write("---")
        coord_mode = st.selectbox("坐标格式", ["Decimal", "CGCS2000"])
        cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
        cm = 0
        if coord_mode == "CGCS2000":
            cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
            
    with col_view:
        preview, final_clean = visualize_lines(st.session_state.cropped_img, line_w, thresh)
        st.image(preview, caption="去线预览 (红线将被删除)", use_column_width=True)

    if st.button("🔥 开始识别", type="primary", use_container_width=True):
        with st.spinner("识别中..."):
            text = pytesseract.image_to_string(final_clean, lang='eng', config='--psm 6')
            st.session_state.raw_text = text
            st.session_state.step = 4
            st.rerun()

# --- 步骤 4: 结果 ---
if st.session_state.step == 4:
    st.divider()
    st.header("4️⃣ 结果生成")
    
    # 动态获取前面选的参数
    # 注意：streamlit在rerun后控件值会重置，这里重新解析一次或依赖session_state
    # 简化逻辑：直接用当前raw_text解析
    
    raw_data = extract_coords(st.session_state.raw_text, coord_mode)
    
    if raw_data:
        df = pd.DataFrame(raw_data)
        edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        
        if st.button("🚀 下载 KMZ"):
            kml = simplekml.Kml()
            for i, row in edited.iterrows():
                try:
                    v1, v2 = float(row["纬度/X"]), float(row["经度/Y"])
                    lat, lon = 0, 0
                    if coord_mode == "Decimal":
                        lat, lon = (v1, v2) if v1 < v2 else (v2, v1)
                    else:
                        res, msg = to_wgs84(v1, v2, cm, False)
                        if res: lat, lon = res, msg
                        else: continue
                    kml.newpoint(name=f"P{i+1}", coords=[(lon, lat)])
                except: continue
            kml.save("out.kmz")
            with open("out.kmz", "rb") as f:
                st.download_button("📥 下载文件", f, "out.kmz", type="primary")
    else:
        st.error("未识别到数据。")
        st.text_area("调试信息", st.session_state.raw_text)
    
    if st.button("🔄 重新开始"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
