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

st.set_page_config(page_title="风资源坐标神器v5.0", page_icon="🧭", layout="centered")

# --- 核心算法 ---
def preprocess_image(image):
    """图像增强：黑白化+强对比度"""
    img = ImageOps.grayscale(image)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.5)
    return img

def dms_to_decimal(d, m, s):
    """度分秒 -> 小数"""
    return float(d) + float(m)/60 + float(s)/3600

def ddm_to_decimal(d, m):
    """度、十进制分 -> 小数 (新功能)"""
    return float(d) + float(m)/60

def extract_all_numbers(text):
    """暴力提取所有数字"""
    # 清洗干扰符
    text = text.replace('°', ' ').replace("'", ' ').replace('"', ' ').replace(':', ' ')
    text = text.replace('l', '1').replace('O', '0').replace('o', '0').replace('|', ' ')
    # 提取浮点数或整数
    nums = re.findall(r"\d+\.\d+|\d+", text)
    return [float(n) for n in nums]

def parse_by_mode(line, mode):
    """根据模式定向抓取"""
    nums = extract_all_numbers(line)
    if not nums: return None, None
    
    # 🎯 模式1：度分秒 (DMS) - 找6个数
    if mode == "DMS":
        if len(nums) < 6: return None, None
        for i in range(len(nums) - 5):
            g = nums[i:i+6]
            # 校验: 度<180, 分<60, 秒<60
            if (g[0]<180 and g[1]<60 and g[2]<60 and 
                g[3]<180 and g[4]<60 and g[5]<60):
                lat = dms_to_decimal(g[0], g[1], g[2])
                lon = dms_to_decimal(g[3], g[4], g[5])
                return lat, lon
    
    # 🎯 模式2：度、十进制分 (DDM) - 找4个数 (新!)
    # 格式: 度 分 度 分 (如 41 12.3456 115 30.1234)
    elif mode == "DDM":
        if len(nums) < 4: return None, None
        for i in range(len(nums) - 3):
            g = nums[i:i+4]
            # 校验: 度<180, 分<60 (分通常带小数)
            if (g[0]<180 and g[1]<60 and g[2]<180 and g[3]<60):
                lat = ddm_to_decimal(g[0], g[1])
                lon = ddm_to_decimal(g[2], g[3])
                return lat, lon

    # 🎯 模式3：小数坐标 (Decimal) - 找2个数
    elif mode == "Decimal":
        # 过滤掉编号(太小)和XY(太大)
        candidates = [n for n in nums if 3 < n < 180]
        if len(candidates) >= 2:
            return candidates[0], candidates[1]

    # 🎯 模式4：大地2000 (CGCS2000) - 找大数
    elif mode == "CGCS2000":
        candidates = [n for n in nums if n > 300000]
        if len(candidates) >= 2:
            return candidates[0], candidates[1]

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
st.title("🧭 风资源坐标神器 v5.0")
st.caption("新增 [度、十进制分] 支持")

with st.sidebar:
    st.header("⚙️ 参数")
    cm_options = {
        "自动(8位带号)": 0, "新疆西 (75)": 75, "新疆中 (81)": 81, "新疆东 (87)": 87,
        "甘肃/青海 (93)": 93, "内蒙西 (99)": 99, "内蒙中 (105)": 105,
        "张家口 (114)": 114, "东北 (123)": 123
    }
    cm_val = cm_options[st.selectbox("大地2000区域", list(cm_options.keys()))]
    force_swap = st.checkbox("强制交换 XY", value=False)

img_file = st.file_uploader("📸 上传图片", type=['png', 'jpg', 'jpeg'])

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["纬度/X", "经度/Y", "原始数据"])

if img_file:
    image = Image.open(img_file)
    st.image(image, caption="已上传", use_column_width=True)
    
    st.write("---")
    st.subheader("🧐 请选择图片里的格式：")
    
    # 这里增加了 DDM 选项
    parse_mode = st.radio(
        "格式类型：",
        ("DMS", "DDM", "Decimal", "CGCS2000"),
        format_func=lambda x: {
            "DMS": "🌐 度 分 秒 (如: 41° 15' 30\")",
            "DDM": "⏱️ 度 十进制分 (如: 41° 15.5')",
            "Decimal": "🔢 纯小数 (如: 41.25833)",
            "CGCS2000": "📐 大地2000 (大数坐标)"
        }[x]
    )

    if st.button('🔥 开始定向识别'):
        processed_img = preprocess_image(image)
        raw_data = []
        with st.spinner('扫描中...'):
            text = pytesseract.image_to_string(processed_img, lang='eng', config='--psm 6')
            lines = text.split('\n')
            for line in lines:
                if not line.strip(): continue
                lat, lon = parse_by_mode(line, parse_mode)
                if lat and lon:
                    if lat > lon and lat < 180: lat, lon = lon, lat
                    raw_data.append({
                        "纬度/X": lat, "经度/Y": lon, 
                        "原始数据": line.strip()[:15]+"..."
                    })
            
            if raw_data:
                st.session_state.df = pd.DataFrame(raw_data)
                st.success(f"✅ 提取到 {len(raw_data)} 行！")
            else:
                st.error(f"❌ 没找到 [{parse_mode}] 格式的数据。请确认选项是否正确。")

st.write("---")
st.subheader("📝 结果核对")
edited_df = st.data_editor(st.session_state.df, num_rows="dynamic")

if st.button("🚀 生成 KMZ"):
    kml = simplekml.Kml()
    cnt = 0
    for idx, row in edited_df.iterrows():
        try:
            v1, v2 = float(row["纬度/X"]), float(row["经度/Y"])
            lat, lon = 0, 0
            
            # 自动判断是否需要转换坐标系
            if v1 < 180 and v2 < 180: 
                lat, lon = v1, v2
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
            st.download_button("📥 下载 KMZ", f, "Coordinates.kmz")
    else:
        st.warning("无数据")
