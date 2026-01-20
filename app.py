import streamlit as st
import simplekml
import re
from pyproj import CRS, Transformer
import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract
import shutil
import pandas as pd
import numpy as np

# --- 环境配置 ---
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    tess_path = shutil.which("tesseract")
    if tess_path: pytesseract.pytesseract.tesseract_cmd = tess_path

st.set_page_config(page_title="风资源坐标神器v6.0", page_icon="🔬", layout="wide")

# --- 核心算法 ---
def dms_to_decimal(d, m, s):
    return float(d) + float(m)/60 + float(s)/3600

def ddm_to_decimal(d, m):
    return float(d) + float(m)/60

def extract_all_numbers(text):
    text = text.replace('°', ' ').replace("'", ' ').replace('"', ' ').replace(':', ' ')
    text = text.replace('l', '1').replace('O', '0').replace('o', '0').replace('|', ' ')
    # 兼容负号
    nums = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", text)
    return [float(n) for n in nums]

def parse_by_mode(line, mode):
    nums = extract_all_numbers(line)
    if not nums: return None, None
    
    if mode == "DMS": # 6参数
        if len(nums) < 6: return None, None
        for i in range(len(nums) - 5):
            g = nums[i:i+6]
            if (abs(g[0])<180 and g[1]<60 and g[2]<60 and 
                abs(g[3])<180 and g[4]<60 and g[5]<60):
                return dms_to_decimal(g[0], g[1], g[2]), dms_to_decimal(g[3], g[4], g[5])
    
    elif mode == "DDM": # 4参数
        if len(nums) < 4: return None, None
        for i in range(len(nums) - 3):
            g = nums[i:i+4]
            if (abs(g[0])<180 and g[1]<60 and abs(g[2])<180 and g[3]<60):
                return ddm_to_decimal(g[0], g[1]), ddm_to_decimal(g[2], g[3])

    elif mode == "Decimal": # 2参数
        candidates = [n for n in nums if 3 < abs(n) < 180]
        if len(candidates) >= 2: return candidates[0], candidates[1]

    elif mode == "CGCS2000": # 大数
        candidates = [n for n in nums if abs(n) > 300000]
        if len(candidates) >= 2: return candidates[0], candidates[1]

    return None, None

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
st.title("🔬 风资源坐标神器 v6.0 (增强画质版)")
st.caption("👈 左侧栏调节【图像处理】，专治模糊/蓝底图片")

# ================= 侧边栏：图像手术台 =================
with st.sidebar:
    st.header("🎛️ 图像预处理 (必看!)")
    st.info("如果识别不出来，请调整这里👇")
    
    # 1. 放大倍数
    zoom_factor = st.slider("🔍 暴力放大 (解决字太小/糊)", 1.0, 4.0, 2.0, 0.5)
    
    # 2. 阈值处理
    st.write("🌗 黑白阈值 (解决蓝底/阴影)")
    threshold_val = st.slider("数值越小越白，越大越黑", 0, 255, 140)
    
    st.write("---")
    st.header("⚙️ 坐标参数")
    cm_options = {
        "自动(8位带号)": 0, "新疆西 (75)": 75, "新疆中 (81)": 81, "新疆东 (87)": 87,
        "甘肃/青海 (93)": 93, "内蒙西 (99)": 99, "内蒙中 (105)": 105,
        "张家口 (114)": 114, "东北 (123)": 123
    }
    cm_val = cm_options[st.selectbox("大地2000区域", list(cm_options.keys()))]
    force_swap = st.checkbox("强制交换 XY", value=False)
# ===================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 上传图片")
    img_file = st.file_uploader("📸 拖入图片", type=['png', 'jpg', 'jpeg'])

processed_img = None

if img_file:
    # 打开原始图片
    original_img = Image.open(img_file)
    
    # --- 图像增强流水线 ---
    # 1. 灰度化
    gray_img = ImageOps.grayscale(original_img)
    
    # 2. 暴力放大 (Resampling.LANCZOS 是抗锯齿最好的算法)
    w, h = gray_img.size
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    resized_img = gray_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 3. 二值化 (手动阈值，去除蓝色背景的关键)
    # 任何亮于 threshold 的像素变白(255)，暗于它的变黑(0)
    fn = lambda x : 255 if x > threshold_val else 0
    binary_img = resized_img.point(fn, mode='1')
    
    processed_img = binary_img

    with col1:
        st.write("🧐 请选择格式：")
        parse_mode = st.radio("格式：", ("Decimal", "DMS", "DDM", "CGCS2000"), horizontal=True)
        
        # 实时预览
        st.image(original_img, caption="原图", use_column_width=True)

    with col2:
        st.subheader("2. 预处理预览 (关键!)")
        st.caption("请调整左侧滑块，直到下图【字是黑的，底是白的】且清晰")
        st.image(processed_img, caption="机器看到的图", use_column_width=True)
        
        if st.button('🔥 这样很清晰了，开始识别!', type="primary"):
            raw_data = []
            with st.spinner('正在玩命扫描...'):
                # 识别参数优化：PSM 6 适合统一的文本块
                text = pytesseract.image_to_string(processed_img, lang='eng', config='--psm 6')
                lines = text.split('\n')
                
                for line in lines:
                    if not line.strip(): continue
                    lat, lon = parse_by_mode(line, parse_mode)
                    if lat and lon:
                        if lat > lon and lat < 180: lat, lon = lon, lat
                        raw_data.append({"纬度/X": lat, "经度/Y": lon, "原文": line[:15]})
            
            if 'df' not in st.session_state: st.session_state.df = pd.DataFrame()
            if raw_data:
                st.session_state.df = pd.DataFrame(raw_data)
                st.success(f"✅ 成功抓取 {len(raw_data)} 行！")
            else:
                st.error("❌ 识别失败。请尝试：\n1. 调节左侧【黑白阈值】滑块\n2. 增加【暴力放大】倍数")

if 'df' in st.session_state and not st.session_state.df.empty:
    st.write("---")
    st.subheader("3. 结果生成")
    edited_df = st.data_editor(st.session_state.df, num_rows="dynamic")
    
    if st.button("🚀 生成 KMZ"):
        kml = simplekml.Kml()
        cnt = 0
        for idx, row in edited_df.iterrows():
            try:
                v1, v2 = float(row["纬度/X"]), float(row["经度/Y"])
                lat, lon = 0, 0
                if v1 < 180 and v2 < 180: lat, lon = v1, v2
                else: 
                    res, msg = cgcs2000_to_wgs84(v1, v2, cm_val, force_swap)
                    if res: lat, lon = res, msg
                    else: continue
                kml.newpoint(name=f"P{idx+1}", coords=[(lon, lat)])
                cnt += 1
            except: continue
        
        if cnt > 0:
            kml.save("final.kmz")
            with open("final.kmz", "rb") as f:
                st.download_button("📥 下载 KMZ", f, "Points.kmz")
