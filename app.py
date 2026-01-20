import streamlit as st
import simplekml
import re
from pyproj import CRS, Transformer
import os
from PIL import Image, ImageOps, ImageEnhance
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

st.set_page_config(page_title="风资源坐标神器v8.0", page_icon="☢️", layout="centered")

# --- 核心算法 ---
def dms_to_decimal(d, m, s):
    return float(d) + float(m)/60 + float(s)/3600

def ddm_to_decimal(d, m):
    return float(d) + float(m)/60

def clean_text_block(text):
    """暴力清洗全文"""
    text = text.replace('|', ' ').replace('!', ' ').replace(']', ' ').replace('[', ' ')
    text = text.replace('°', ' ').replace("'", ' ').replace('"', ' ').replace(':', ' ')
    text = text.replace('l', '1').replace('O', '0').replace('o', '0')
    return text

def extract_numbers_from_text(text):
    """从文本中提取所有浮点数"""
    # 兼容负号
    nums = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", text)
    return [float(n) for n in nums]

def global_harvest(text, mode):
    """☢️ 核弹模式：全图数字暴力配对"""
    # 1. 提取全图所有数字
    clean_txt = clean_text_block(text)
    all_nums = extract_numbers_from_text(clean_txt)
    
    pairs = []
    
    if mode == "Decimal":
        # 过滤：只留像坐标的数 (3 < x < 180)
        valid_nums = [n for n in all_nums if 3 < abs(n) < 180]
        # 强制两两配对
        # 假设顺序是: Lat, Lon, Lat, Lon...
        for i in range(0, len(valid_nums) - 1, 2):
            pairs.append((valid_nums[i], valid_nums[i+1]))
            
    elif mode == "CGCS2000":
        # 过滤：只留大数 (> 300,000)
        valid_nums = [n for n in all_nums if abs(n) > 300000]
        for i in range(0, len(valid_nums) - 1, 2):
            pairs.append((valid_nums[i], valid_nums[i+1]))
            
    # DMS 和 DDM 比较复杂，全图扫描容易乱序，暂时依赖行扫描
    # 但如果用户选了DMS/DDM且行扫描失败，我们也可以尝试找符合逻辑的组
    
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
st.title("☢️ 风资源坐标神器 v8.0")
st.caption("新增【暴力收割模式】，专治表格识别失败")

img_file = st.file_uploader("📄 请先上传图片", type=['png', 'jpg', 'jpeg'])

if img_file:
    st.divider()
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. 图像预处理")
        # 你的图片需要较高的阈值来去除蓝色背景
        thresh = st.slider("🌗 黑白阈值 (往左拖去背景)", 0, 255, 110, help="针对蓝底图片，试着调低这个值")
        zoom = st.slider("🔎 暴力放大", 1.0, 4.0, 2.0)
        
        image = Image.open(img_file)
        gray = ImageOps.grayscale(image)
        w, h = gray.size
        resized = gray.resize((int(w * zoom), int(h * zoom)), Image.Resampling.LANCZOS)
        fn = lambda x : 255 if x > thresh else 0
        processed_img = resized.point(fn, mode='1')
        
        st.image(processed_img, caption="机器看到的图 (字一定要黑，底一定要白)", use_column_width=True)

    with col2:
        st.subheader("2. 识别设置")
        mode = st.radio("坐标格式", ("Decimal", "DMS", "DDM", "CGCS2000"), 
                 format_func=lambda x: {
                     "Decimal": "🔢 纯小数 (如 82.7807)", 
                     "DMS": "🌐 度分秒 (如 41°15'30\")",
                     "DDM": "⏱️ 度+分 (如 41°15.5')",
                     "CGCS2000": "📐 大地2000 (大数)"
                 }[x])
        
        # 动态显示设置
        cm_val = 0
        force_swap = False
        if mode == "CGCS2000":
            st.info("设置大地2000参数：")
            cm_options = {"自动(8位带号)": 0, "75": 75, "81": 81, "87": 87, "93": 93, "99": 99, "105": 105, "114": 114, "123": 123}
            cm_val = cm_options[st.selectbox("中央经线", list(cm_options.keys()))]
            force_swap = st.checkbox("强制交换 XY")

        st.write("")
        if st.button("🔥 开始识别", type="primary", use_container_width=True):
            with st.spinner("正在进行核弹级扫描..."):
                # 获取原始文本
                raw_text = pytesseract.image_to_string(processed_img, lang='eng', config='--psm 6')
                
                raw_data = []
                method_used = "按行精准扫描"
                
                # 1. 尝试按行扫描 (保留行信息)
                lines = raw_text.split('\n')
                for line in lines:
                    if not line.strip(): continue
                    clean_line = clean_text_block(line)
                    nums = extract_numbers_from_text(clean_line)
                    
                    # 简单的按行逻辑
                    lat, lon = None, None
                    if mode == "Decimal":
                        cands = [n for n in nums if 3 < abs(n) < 180]
                        if len(cands) >= 2: lat, lon = cands[0], cands[1]
                    elif mode == "DMS" and len(nums) >= 6:
                         # 简化逻辑，只取前6个
                         lat = dms_to_decimal(nums[0], nums[1], nums[2])
                         lon = dms_to_decimal(nums[3], nums[4], nums[5])
                    
                    if lat and lon:
                        if lat > lon and lat < 180: lat, lon = lon, lat
                        raw_data.append({"纬度/X": lat, "经度/Y": lon, "来源": "行扫描"})

                # 2. 如果按行扫描失败，启动【暴力收割】
                if not raw_data and mode in ["Decimal", "CGCS2000"]:
                    method_used = "☢️ 暴力收割模式"
                    pairs = global_harvest(raw_text, mode)
                    for p in pairs:
                        lat, lon = p[0], p[1]
                        if lat > lon and lat < 180: lat, lon = lon, lat
                        raw_data.append({"纬度/X": lat, "经度/Y": lon, "来源": "暴力收割"})

                # 结果展示
                if raw_data:
                    st.session_state.df = pd.DataFrame(raw_data)
                    if method_used == "☢️ 暴力收割模式":
                        st.warning("⚠️ 按行识别失败，已自动切换为【暴力收割模式】！程序忽略了表格线，强行提取了全图数字并配对。请务必检查顺序是否正确。")
                    else:
                        st.success(f"✅ 成功提取 {len(raw_data)} 行！")
                else:
                    st.error("❌ 识别彻底失败。")
                    with st.expander("👀 点这里查看机器读到了什么 (RAW)"):
                        st.text(raw_text)
                        st.caption("如果上面是空的或乱码，说明图像预处理没弄好，请调节左侧滑块。")

    # 3. 结果生成
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
                    if v1 > 180 or v2 > 180: # 大数才转
                         res, msg = cgcs2000_to_wgs84(v1, v2, cm_val, force_swap)
                         if res: lat, lon = res, msg
                         else: continue
                    else: lat, lon = v1, v2
                    kml.newpoint(name=f"P{idx+1}", coords=[(lon, lat)])
                except: continue
            kml.save("points.kmz")
            with open("points.kmz", "rb") as f:
                st.download_button("📥 下载 KMZ", f, "Points.kmz", type="primary")
