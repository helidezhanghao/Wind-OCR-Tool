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

st.set_page_config(page_title="力力的坐标工具 v18.0", page_icon="📍", layout="centered")

# --- 状态初始化 ---
if 'angle' not in st.session_state:
    st.session_state.angle = 0.0
if 'raw_img' not in st.session_state:
    st.session_state.raw_img = None
if 'final_img' not in st.session_state:
    st.session_state.final_img = None
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = ""
if 'parsed_df' not in st.session_state:
    st.session_state.parsed_df = None

# --- 核心工具函数 ---
def rotate_image(image, angle):
    return image.rotate(angle, expand=True)

def simple_preprocess(pil_image, threshold):
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

def extract_data_from_lines(text, mode):
    """
    按行解析，提取 编号 + 坐标
    """
    lines = text.split('\n')
    data = []
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # 预处理行：去除干扰字符
        clean_line = line.replace('|', ' ').replace('[', ' ').replace(']', ' ')
        clean_line = clean_line.replace('°', ' ').replace("'", ' ').replace('"', ' ').replace(':', ' ')
        
        # 提取该行所有数字
        raw_nums = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", clean_line)
        nums_val = [float(n) for n in raw_nums]
        
        # 提取该行所有文本片段（用来找编号）
        parts = clean_line.split()
        if not parts: continue
        
        # 假设第一个片段是编号
        potential_id = parts[0]
        
        lat, lon = None, None
        
        # --- 模式匹配逻辑 ---
        if mode == "Decimal":
            # 找 3 < x < 180 的数字作为坐标
            coords = [smart_fix_decimal(n) for n in nums_val if 3 < abs(n) < 180]
            if len(coords) >= 2:
                lat, lon = coords[0], coords[1]
                
        elif mode == "DMS": 
            # 需要6个数字
            if len(nums_val) >= 6:
                # 简单逻辑：取前6个
                g = nums_val[:6]
                if (abs(g[0])<180 and g[1]<60 and g[2]<60 and 
                    abs(g[3])<180 and g[4]<60 and g[5]<60):
                    lat = dms_to_dec(g[0], g[1], g[2])
                    lon = dms_to_dec(g[3], g[4], g[5])

        elif mode == "DDM": 
            # 需要4个数字
            if len(nums_val) >= 4:
                g = nums_val[:4]
                if (abs(g[0])<180 and g[1]<60 and abs(g[2])<180 and g[3]<60):
                    lat = ddm_to_dec(g[0], g[1])
                    lon = ddm_to_dec(g[2], g[3])

        elif mode == "CGCS2000":
            # 找大数
            coords = [n for n in nums_val if abs(n) > 300000]
            if len(coords) >= 2:
                lat, lon = coords[0], coords[1]
        
        # --- 组装数据 ---
        if lat is not None and lon is not None:
            # 如果提取到的编号长得像坐标数字，说明可能这一行没编号，是纯数字
            # 简单判断：如果 potential_id 转成数字后等于 lat 或 lon，说明它不是编号
            try:
                if abs(float(potential_id) - lat) < 0.001 or abs(float(potential_id) - lon) < 0.001:
                    row_id = "Auto" # 没找到独立编号
                else:
                    row_id = potential_id
            except:
                row_id = potential_id # 转不成数字，肯定是编号
            
            data.append({"编号": row_id, "纬度/X": lat, "经度/Y": lon})
            
    return data

def to_wgs84(v1, v2, cm, swap):
    x, y = (v2, v1) if swap else (v1, v2)
    # 防止代码过长报错
    if 10000000 < x < 100000000 and y < 10000000: 
        x, y = y, x
    
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

st.title("📍 力力的坐标工具 v18.0")

# --- 步骤 1: 上传 ---
st.header("1. 上传图片")
img_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

if img_file:
    if 'last_file' not in st.session_state or st.session_state.last_file != img_file.name:
        st.session_state.last_file = img_file.name
        st.session_state.raw_img = Image.open(img_file)
        st.session_state.angle = 0.0
        st.session_state.final_img = st.session_state.raw_img 
        st.session_state.raw_text = ""
        st.session_state.parsed_df = None
        st.rerun()

if st.session_state.raw_img:
    st.divider()
    st.header("2. 图像处理 (可选)")
    
    enable_crop = st.checkbox("✂️ 需要旋转或裁切？", value=False)
    
    if enable_crop:
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
            input_angle = st.number_input("微调角度", value=float(st.session_state.angle), step=0.5)
            if input_angle != st.session_state.angle:
                st.session_state.angle = input_angle
                st.rerun()

        rotated = rotate_image(st.session_state.raw_img, st.session_state.angle)
        st.caption("👇 拖动红框选中数据区域：")
        
        cropped_out = st_cropper(rotated, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
        st.session_state.final_img = cropped_out
        st.divider()
    else:
        st.session_state.final_img = st.session_state.raw_img
    
    # --- 步骤 3: 识别设置 ---
    st.header("3. 识别参数")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚙️ 设置")
        
        coord_options = {
            "Decimal": "🔢 纯小数 (82.78)",
            "DMS": "🌐 度分秒 (41°15'30\")",
            "DDM": "⏱️ 度+分 (41°15.5')",
            "CGCS2000": "📐 大地2000"
        }
        
        coord_mode = st.selectbox("坐标格式", 
                                  list(coord_options.keys()),
                                  format_func=lambda x: coord_options[x])
        
        cm_ops = {
            0: 0, 
            75: 75, 81: 81, 87: 87, 
            93: 93, 99: 99, 105: 105, 
            114: 114, 123: 123
        }
        
        cm = 0
        if coord_mode == "CGCS2000":
            cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
            
        st.write("---")
        thresh = st.slider("黑白阈值 (看不清字就调这个)", 0, 255, 120)

    with col2:
        st.subheader("👀 预览")
        processed_preview = simple_preprocess(st.session_state.final_img, thresh)
        st.image(processed_preview, caption="最终识别图", use_column_width=True)

    st.write("")
    if st.button("🔥 提取原始文本", type="primary", use_container_width=True):
        with st.spinner("提取中..."):
            final_processed = simple_preprocess(st.session_state.final_img, thresh)
            text = pytesseract.image_to_string(final_processed, lang='eng', config='--psm 6')
            st.session_state.raw_text = text
            # 清除旧的表格数据
            st.session_state.parsed_df = None

    # --- 步骤 4: 确认与修改 ---
    if st.session_state.raw_text:
        st.divider()
        st.header("4. 确认与编辑原始数据")
        st.info("👇 请检查下方的识别结果。如果编号错了，或者数据换行了，请直接在这里修改！")
        
        # 让用户修改 OCR 结果
        edited_text = st.text_area("OCR 原始结果 (每一行代表一组数据)", 
                                   value=st.session_state.raw_text, 
                                   height=200)
        
        if st.button("⚡ 解析表格数据", use_container_width=True):
            raw_data = extract_data_from_lines(edited_text, coord_mode)
            if raw_data:
                st.session_state.parsed_df = pd.DataFrame(raw_data)
            else:
                st.error("无法从文本中解析出坐标，请检查文本格式。")

    # --- 步骤 5: 结果展示与生成 ---
    if st.session_state.parsed_df is not None:
        st.divider()
        st.header("5. 结果核对与生成")
        
        # 显示可编辑的表格
        final_df = st.data_editor(st.session_state.parsed_df, num_rows="dynamic", use_container_width=True)
        
        if st.button("🚀 生成 KMZ"):
            kml = simplekml.Kml()
            valid_count = 0
            for i, row in final_df.iterrows():
                try:
                    v1 = float(row["纬度/X"])
                    v2 = float(row["经度/Y"])
                    # 使用提取到的编号，如果没有则用行号
                    pt_name = str(row.get("编号", f"P{i+1}"))
                    
                    lat, lon = 0, 0
                    
                    if coord_mode in ["Decimal", "DMS", "DDM"]:
                        lat, lon = (v1, v2) if v1 < v2 else (v2, v1)
                    else:
                        res, msg = to_wgs84(v1, v2, cm, False)
                        if res: lat, lon = res, msg
                        else: continue
                    
                    kml.newpoint(name=pt_name, coords=[(lon, lat)])
                    valid_count += 1
                except: continue
            
            if valid_count > 0:
                kml.save("out.kmz")
                with open("out.kmz", "rb") as f:
                    st.download_button("📥 下载 KMZ 文件", f, "out.kmz", type="primary")
            else:
                st.warning("没有有效的坐标点生成。")
