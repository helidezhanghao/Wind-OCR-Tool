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
import cv2  # 引入工业级视觉库

# --- 环境配置 ---
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    tess_path = shutil.which("tesseract")
    if tess_path: pytesseract.pytesseract.tesseract_cmd = tess_path

st.set_page_config(page_title="风资源坐标神器v9.0", page_icon="🧿", layout="centered")

# --- 核心算法 ---
def dms_to_decimal(d, m, s):
    return float(d) + float(m)/60 + float(s)/3600

def ddm_to_decimal(d, m):
    return float(d) + float(m)/60

def clean_text_block(text):
    # 极度暴力的清洗，把可能干扰数字的符号全换空格
    text = text.replace('|', ' ').replace('!', ' ').replace(']', ' ').replace('[', ' ')
    text = text.replace('°', ' ').replace("'", ' ').replace('"', ' ').replace(':', ' ')
    text = text.replace('l', '1').replace('O', '0').replace('o', '0')
    # 去除常见的 T1, T2 编号干扰 (把 T 换成空格)
    text = text.replace('T', ' ').replace('t', ' ')
    return text

def extract_numbers_from_text(text):
    # 提取所有数字
    nums = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", text)
    return [float(n) for n in nums]

def process_image_opencv(pil_image, zoom, remove_grid):
    """工业级图像处理：自适应阈值 + 形态学去线"""
    # 1. 转为 OpenCV 格式
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # 2. 暴力放大
    h, w = img_cv.shape[:2]
    img_cv = cv2.resize(img_cv, (int(w*zoom), int(h*zoom)), interpolation=cv2.INTER_CUBIC)
    
    # 3. 转灰度
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 4. 自适应二值化 (关键！专治光照不均/蓝底)
    # block_size 决定了局部区域的大小，C 是常数
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 31, 15)
    
    # 5. 去除表格线 (可选)
    if remove_grid:
        # 定义横线和竖线结构
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # 检测线
        hor_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hor_kernel, iterations=2)
        ver_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ver_kernel, iterations=2)
        
        # 这种方法是把检测到的线“加粗”然后变成白色(背景)，从而抹除黑色线条
        # 更好的方法是：用原图减去线条图？或者直接把线条区域填白
        # 这里用简单的 mask 填白
        cnts_h, _ = cv2.findContours(hor_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_v, _ = cv2.findContours(ver_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 把线涂白
        cv2.drawContours(binary, cnts_h, -1, (255,255,255), 5)
        cv2.drawContours(binary, cnts_v, -1, (255,255,255), 3)

    return Image.fromarray(binary)

def global_harvest(text, mode):
    """☢️ 暴力收割 V2"""
    clean_txt = clean_text_block(text)
    all_nums = extract_numbers_from_text(clean_txt)
    pairs = []
    
    if mode == "Decimal":
        # 过滤掉编号(通常<30或整数)，保留像坐标的数(30 < x < 180)
        # 你的图里是 82.xxx 和 43.xxx，所以阈值设为 30 比较稳
        valid_nums = [n for n in all_nums if 20 < abs(n) < 180]
        # 强制配对
        for i in range(0, len(valid_nums) - 1, 2):
            pairs.append((valid_nums[i], valid_nums[i+1]))
            
    elif mode == "CGCS2000":
        valid_nums = [n for n in all_nums if abs(n) > 300000]
        for i in range(0, len(valid_nums) - 1, 2):
            pairs.append((valid_nums[i], valid_nums[i+1]))
            
    return pairs

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
st.title("🧿 风资源坐标神器 v9.0 (工业版)")
st.caption("引入 OpenCV 自适应处理 · 专治蓝底烂图")

img_file = st.file_uploader("📄 上传图片", type=['png', 'jpg', 'jpeg'])

if img_file:
    st.divider()
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. 图像增强")
        # 默认开启去网格
        remove_grid = st.checkbox("🔪 自动抹除表格线 (推荐开启)", value=True)
        zoom = st.slider("🔎 放大倍数", 1.5, 3.5, 2.0)
        
        image = Image.open(img_file)
        # 调用 OpenCV 处理
        processed_img = process_image_opencv(image, zoom, remove_grid)
        
        st.image(processed_img, caption="机器眼中的画面 (注意看左侧数字是否清晰)", use_column_width=True)

    with col2:
        st.subheader("2. 识别设置")
        mode = st.radio("坐标格式", ("Decimal", "DMS", "DDM", "CGCS2000"), 
                 format_func=lambda x: {
                     "Decimal": "🔢 纯小数 (如 82.7807)", 
                     "DMS": "🌐 度分秒 (如 41°15'30\")",
                     "DDM": "⏱️ 度+分 (如 41°15.5')",
                     "CGCS2000": "📐 大地2000 (大数)"
                 }[x])
        
        cm_val = 0
        force_swap = False
        if mode == "CGCS2000":
            st.info("设置大地2000参数：")
            cm_options = {"自动(8位带号)": 0, "75": 75, "81": 81, "87": 87, "93": 93, "99": 99, "105": 105, "114": 114, "123": 123}
            cm_val = cm_options[st.selectbox("中央经线", list(cm_options.keys()))]
            force_swap = st.checkbox("强制交换 XY")

        st.write("")
        if st.button("🔥 开始识别", type="primary", use_container_width=True):
            with st.spinner("正在进行工业级扫描..."):
                raw_text = pytesseract.image_to_string(processed_img, lang='eng', config='--psm 6')
                
                raw_data = []
                # 直接使用暴力收割模式，因为对于这种图，按行识别太容易受干扰
                pairs = global_harvest(raw_text, mode)
                
                for p in pairs:
                    lat, lon = p[0], p[1]
                    # 自动归位：在中国，经度(73-135) > 纬度(18-54)
                    # 你图里是 82(经) 和 43(纬)
                    if lat > lon and lat < 180: lat, lon = lon, lat # 确保lat是小的，lon是大的
                    # 再次校验，如果反了（比如lon是82，lat是43，上面逻辑会变成 lat=43, lon=82，这是对的）
                    # 但如果本来就是 lat=82(国外?), lon=43，这个逻辑会强制把大的当经度。
                    # 针对你的图：T1 82... 43... -> 82是经度，43是纬度。
                    # 结果应为: 纬度43, 经度82
                    if lat > 60: # 简单的中国区判断，纬度很少超过60
                         lat, lon = lon, lat

                    raw_data.append({"纬度/X": lat, "经度/Y": lon, "来源": "暴力收割"})

                if raw_data:
                    st.session_state.df = pd.DataFrame(raw_data)
                    st.success(f"✅ 成功提取 {len(raw_data)} 行！")
                else:
                    st.error("❌ 识别失败。")
                    with st.expander("调试信息"):
                        st.text(raw_text)

    if 'df' in st.session_state and not st.session_state.df.empty:
        st.divider()
        st.subheader("3. 结果核对")
        edited_df = st.data_editor(st.session_state.df, num_rows="dynamic", use_container_width=True)
        
        if st.button("🚀 生成 KMZ"):
            kml = simplekml.Kml()
            for idx, row in edited_df.iterrows():
                try:
                    v1, v2 = float(row["纬度/X"]), float(row["经度/Y"])
                    lat, lon = 0, 0
                    if v1 > 180 or v2 > 180:
                         res, msg = cgcs2000_to_wgs84(v1, v2, cm_val, force_swap)
                         if res: lat, lon = res, msg
                         else: continue
                    else: lat, lon = v1, v2
                    kml.newpoint(name=f"P{idx+1}", coords=[(lon, lat)])
                except: continue
            kml.save("points.kmz")
            with open("points.kmz", "rb") as f:
                st.download_button("📥 下载 KMZ", f, "Points.kmz", type="primary")
