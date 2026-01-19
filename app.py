import streamlit as st
import simplekml
import re
from pyproj import CRS, Transformer
import os
from PIL import Image
import pytesseract
import shutil
import numpy as np

# ================= 核心配置：自动适配云端环境 =================
# Streamlit Cloud 是 Linux 系统，不需要像 Windows 那样指定 C 盘路径
# 我们让系统自动去找 Tesseract 在哪里
tess_path = shutil.which("tesseract")
if tess_path:
    pytesseract.pytesseract.tesseract_cmd = tess_path
else:
    # 如果是本地 Windows 测试（兼容代码）
    if os.name == 'nt':
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# ==========================================================

st.set_page_config(page_title="风资源坐标通", page_icon="🌍")
st.title("🌍 风资源坐标神器 (手机云端版)")
st.info("💡 即使电脑关机，手机也能随时访问使用！")

# --- 坐标转换逻辑 (保持不变) ---
def dms_to_dd(dms_str):
    try:
        parts = re.findall(r"[\d\.]+", dms_str)
        if len(parts) < 3: return float(parts[0])
        d, m, s = float(parts[0]), float(parts[1]), float(parts[2])
        return d + m/60 + s/3600
    except: return 0

def parse_line(line):
    # 清洗 OCR 产生的杂质
    clean_line = line.replace("|", " ").replace("[", "").replace("]", "").replace("X", "").replace("Y", "").replace("=", "").replace("：", "")
    clean_line = clean_line.replace("\u3000", " ").strip()
    # 修复常见 OCR 错误 (l->1, O->0)
    clean_line = clean_line.replace('l', '1').replace('O', '0').replace('o', '0')
    
    parts = re.split(r"[,，\s]+", clean_line)
    parts = [p for p in parts if p]
    
    if len(parts) < 2: return None, "格式不足"
    v1_str, v2_str = parts[0], parts[1]
    
    if "°" in line or "'" in line:
        return (dms_to_dd(v1_str), dms_to_dd(v2_str)), "DMS"
    try:
        v1, v2 = float(v1_str), float(v2_str)
        if v1 < 180 and v2 < 180: return (v1, v2), "WGS84"
        else: return (v1, v2), "CGCS2000"
    except: return None, "非数字"

def cgcs2000_to_wgs84(v1, v2, cm_val, force_swap):
    val_a, val_b = v1, v2
    x_coord, y_coord = 0, 0
    if force_swap: y_coord, x_coord = val_a, val_b
    else:
        s_a, s_b = str(int(val_a)), str(int(val_b))
        if len(s_a) == 7 and (len(s_b) == 8 or len(s_b) == 6): x_coord, y_coord = val_a, val_b
        elif len(s_b) == 7 and (len(s_a) == 8 or len(s_a) == 6): x_coord, y_coord = val_b, val_a
        else: x_coord, y_coord = val_a, val_b

    y_str = str(int(y_coord))
    final_cm = 0
    if len(y_str) == 8: final_cm = int(y_str[:2]) * 3
    else:
        if cm_val == 0: return None, "6位坐标需选区域"
        final_cm = cm_val

    false_easting = 500000 + (int(y_str[:2]) * 1000000 if len(y_str) == 8 else 0)
    crs_str = f"+proj=tmerc +lat_0=0 +lon_0={final_cm} +k=1 +x_0={false_easting} +y_0=0 +ellps=GRS80 +units=m +no_defs"
    
    try:
        transformer = Transformer.from_crs(CRS.from_string(crs_str), CRS.from_epsg(4326), always_xy=True)
        lon, lat = transformer.transform(y_coord, x_coord)
        return (lat, lon), "OK"
    except Exception as e: return None, str(e)

# --- 网页界面 ---
with st.sidebar:
    st.header("⚙️ 设置")
    cm_options = {
        "自动(8位带号)": 0, "75 (新疆西)": 75, "81 (新疆中)": 81, "87 (新疆东)": 87,
        "93 (甘肃/青海)": 93, "99 (内蒙西)": 99, "105 (内蒙中)": 105,
        "111 (晋/陕)": 111, "114 (张家口)": 114, "117 (京/承)": 117,
        "120 (鲁/内蒙东)": 120, "123 (东北)": 123
    }
    selected_cm_label = st.selectbox("区域/中央经线", list(cm_options.keys()))
    selected_cm_val = cm_options[selected_cm_label]
    force_swap = st.checkbox("强制交换 XY", value=False)

st.write("👇 上传照片或直接粘贴文本")
img_file = st.file_uploader("📸 拍照上传", type=['png', 'jpg', 'jpeg'])
manual_text = st.text_area("✍️ 粘贴文本", height=100)

input_data = ""
if img_file:
    image = Image.open(img_file)
    st.image(image, caption='已上传', use_column_width=True)
    if st.button('🔍 识别文字'):
        with st.spinner('正在云端识别...'):
            try:
                # 关键：config参数优化表格识别
                text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
                st.success("识别完成！")
                input_data = text
            except Exception as e:
                st.error(f"识别失败: {e}")

final_text = manual_text if manual_text else input_data

if final_text:
    st.text_area("结果预览 (请手动修正)", value=final_text, height=150, key="editor")
    if st.button("🚀 生成 KMZ"):
        lines = final_text.split('\n')
        kml = simplekml.Kml()
        valid = 0
        logs = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            res, type_ = parse_line(line)
            if not res: continue
            lat, lon = 0, 0
            success = False
            if type_ in ["WGS84", "DMS"]:
                v1, v2 = res
                lat, lon = (v1, v2) if v1 < v2 else (v2, v1)
                success = True
                logs.append(f"✅ 行{i+1}: 经纬度 {lat:.4f}, {lon:.4f}")
            elif type_ == "CGCS2000":
                wgs, msg = cgcs2000_to_wgs84(res[0], res[1], selected_cm_val, force_swap)
                if wgs:
                    lat, lon = wgs
                    success = True
                    logs.append(f"✅ 行{i+1}: 转换成功")
                else: logs.append(f"❌ 行{i+1}: {msg}")
            
            if success:
                kml.newpoint(name=f"P{valid+1}", coords=[(lon, lat)])
                valid += 1
        
        st.write("\n".join(logs))
        if valid > 0:
            kml.save("out.kmz")
            with open("out.kmz", "rb") as f:
                st.download_button("📥 下载 KMZ", f, file_name="Points.kmz")
        else: st.warning("无有效坐标")