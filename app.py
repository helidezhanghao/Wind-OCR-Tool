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

st.set_page_config(page_title="力力的坐标工具 v19.2", page_icon="📍", layout="centered")

# --- 通用工具函数 ---

def parse_dms_string(s):
    """解析Excel中的度分秒字符串 (如 57° 56' 22.39" E)"""
    s_str = str(s).upper()
    # 简单的清洗
    clean = s_str.replace('°', ' ').replace("'", ' ').replace('"', ' ').replace(':', ' ')
    # 提取数字
    parts = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", clean)
    if len(parts) >= 3:
        d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        val = d + m/60 + sec/3600
        # 简单判断南纬西经
        if 'S' in s_str or 'W' in s_str: val = -val
        return val
    return 0.0

def parse_ddm_string(s):
    """解析Excel中的度分字符串"""
    s_str = str(s).upper()
    clean = s_str.replace('°', ' ').replace("'", ' ').replace(':', ' ')
    parts = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", clean)
    if len(parts) >= 2:
        d, m = float(parts[0]), float(parts[1])
        val = d + m/60
        if 'S' in s_str or 'W' in s_str: val = -val
        return val
    return 0.0

def to_wgs84(v1, v2, cm, swap):
    x, y = (v2, v1) if swap else (v1, v2)
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

def generate_kmz(df, coord_mode, cm=0):
    """
    通用生成KMZ逻辑，支持Excel导入的字符串解析
    """
    kml = simplekml.Kml()
    valid_count = 0
    for i, row in df.iterrows():
        try:
            raw_v1 = row.get("纬度/X", row.get("Lat", 0))
            raw_v2 = row.get("经度/Y", row.get("Lon", 0))
            name = str(row.get("编号", f"P{i+1}"))
            
            v1, v2 = 0.0, 0.0
            
            # --- 核心修复：根据模式解析 ---
            if coord_mode == "Decimal":
                v1, v2 = float(raw_v1), float(raw_v2)
            elif coord_mode == "DMS":
                v1 = parse_dms_string(raw_v1)
                v2 = parse_dms_string(raw_v2)
            elif coord_mode == "DDM":
                v1 = parse_ddm_string(raw_v1)
                v2 = parse_ddm_string(raw_v2)
            elif coord_mode == "CGCS2000":
                v1, v2 = float(raw_v1), float(raw_v2)
            
            lat, lon = 0, 0
            if coord_mode != "CGCS2000":
                lat, lon = (v1, v2) if v1 < v2 else (v2, v1)
            else:
                res, msg = to_wgs84(v1, v2, cm, False)
                if res: lat, lon = res, msg
                else: continue
            
            if abs(lat) > 0.1 and abs(lon) > 0.1:
                kml.newpoint(name=name, coords=[(lon, lat)])
                valid_count += 1
        except: continue
    return kml, valid_count

# --- 图片识别专用工具 ---
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

def extract_img_dms(d, m, s):
    return float(d) + float(m)/60 + float(s)/3600

def extract_img_ddm(d, m):
    return float(d) + float(m)/60

def extract_data_from_text(text, mode):
    lines = text.split('\n')
    data = []
    for line in lines:
        line = line.strip()
        if not line: continue
        clean_line = line.replace('|', ' ').replace('[', ' ').replace(']', ' ').replace('°', ' ').replace("'", ' ').replace('"', ' ').replace(':', ' ')
        raw_nums = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", clean_line)
        nums_val = [float(n) for n in raw_nums]
        parts = clean_line.split()
        if not parts: continue
        potential_id = parts[0]
        
        lat, lon = None, None
        if mode == "Decimal":
            coords = [smart_fix_decimal(n) for n in nums_val if 3 < abs(n) < 180]
            if len(coords) >= 2: lat, lon = coords[0], coords[1]
        elif mode == "DMS" and len(nums_val) >= 6:
            g = nums_val[:6]
            if abs(g[0])<180 and g[1]<60 and g[2]<60:
                lat = extract_img_dms(g[0], g[1], g[2])
                lon = extract_img_dms(g[3], g[4], g[5])
        elif mode == "DDM" and len(nums_val) >= 4:
            g = nums_val[:4]
            if abs(g[0])<180 and g[1]<60:
                lat = extract_img_ddm(g[0], g[1])
                lon = extract_img_ddm(g[2], g[3])
        elif mode == "CGCS2000":
            coords = [n for n in nums_val if abs(n) > 300000]
            if len(coords) >= 2: lat, lon = coords[0], coords[1]
        
        if lat is not None and lon is not None:
            try:
                if abs(float(potential_id) - lat) < 0.001 or abs(float(potential_id) - lon) < 0.001:
                    row_id = "Auto"
                else:
                    row_id = potential_id
            except: row_id = potential_id
            data.append({"编号": row_id, "纬度/X": lat, "经度/Y": lon})
    return data

# ================= 界面主逻辑 =================

st.title("📍 力力的坐标工具 v19.2")

with st.sidebar:
    st.header("功能选择")
    app_mode = st.radio("请选择使用模式：", 
                        ["🖐️ 手动输入", "📊 Excel表格识别", "📸 图片识别"],
                        index=2)
    st.divider()
    st.info("切换模式会清空当前数据")

# ==========================================
# 模式 1：手动输入
# ==========================================
if app_mode == "🖐️ 手动输入":
    st.header("🖐️ 手动录入坐标")
    
    col1, col2 = st.columns(2)
    with col1:
        # 修正：补全所有选项
        coord_mode = st.selectbox("坐标格式", 
                                  ["Decimal", "DMS", "DDM", "CGCS2000"],
                                  format_func=lambda x: {
                                      "Decimal": "🔢 纯小数",
                                      "DMS": "🌐 度分秒",
                                      "DDM": "⏱️ 度+分",
                                      "CGCS2000": "📐 大地2000"
                                  }[x])
    cm = 0
    with col2:
        if coord_mode == "CGCS2000":
            cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
            cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
    
    st.subheader("📝 在下方表格输入数据")
    if coord_mode in ["DMS", "DDM"]:
        st.info("支持输入格式如：57° 56' 22.39\"")

    if 'manual_df' not in st.session_state:
        st.session_state.manual_df = pd.DataFrame([
            {"编号": "T1", "纬度/X": "", "经度/Y": ""},
            {"编号": "T2", "纬度/X": "", "经度/Y": ""},
        ])
    
    edited_df = st.data_editor(st.session_state.manual_df, num_rows="dynamic", use_container_width=True)
    
    if st.button("🚀 生成 KMZ", type="primary"):
        kml, count = generate_kmz(edited_df, coord_mode, cm)
        if count > 0:
            kml.save("manual.kmz")
            with open("manual.kmz", "rb") as f:
                st.download_button("📥 下载文件", f, "manual.kmz")
        else:
            st.error("无效数据，请检查输入格式。")

# ==========================================
# 模式 2：Excel 表格识别
# ==========================================
elif app_mode == "📊 Excel表格识别":
    st.header("📊 Excel 批量导入")
    
    excel_file = st.file_uploader("上传 Excel 文件 (.xlsx, .xls)", type=['xlsx', 'xls'])
    
    if excel_file:
        try:
            df = pd.read_excel(excel_file)
            st.success("读取成功！")
            
            st.write("### 1. 数据映射")
            cols = list(df.columns)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                col_name = st.selectbox("编号列 (可选)", ["无"] + cols)
            with c2:
                default_lat = next((c for c in cols if "纬" in c or "Lat" in c or "X" in c or "北" in c), cols[0])
                col_lat = st.selectbox("纬度 / X坐标 列", cols, index=cols.index(default_lat) if default_lat in cols else 0)
            with c3:
                default_lon = next((c for c in cols if "经" in c or "Lon" in c or "Y" in c or "东" in c), cols[0])
                col_lon = st.selectbox("经度 / Y坐标 列", cols, index=cols.index(default_lon) if default_lon in cols else 0)
            
            processed_data = []
            for i, row in df.iterrows():
                processed_data.append({
                    "编号": row[col_name] if col_name != "无" else f"P{i+1}",
                    "纬度/X": row[col_lat],
                    "经度/Y": row[col_lon]
                })
            processed_df = pd.DataFrame(processed_data)
            
            st.write("### 2. 确认与生成")
            
            col_set1, col_set2 = st.columns(2)
            with col_set1:
                # 🔥 修正：这里补全了所有选项！！
                coord_mode = st.selectbox("Excel中的坐标格式", 
                                          ["Decimal", "DMS", "DDM", "CGCS2000"],
                                          format_func=lambda x: {
                                              "Decimal": "🔢 纯小数 (如 82.78)",
                                              "DMS": "🌐 度分秒 (如 57°56'22\")",
                                              "DDM": "⏱️ 度+分 (如 41°15.5')",
                                              "CGCS2000": "📐 大地2000"
                                          }[x])
            cm = 0
            with col_set2:
                if coord_mode == "CGCS2000":
                    cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                    cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
            
            st.caption("👇 数据预览 (可修改)：")
            final_df = st.data_editor(processed_df, num_rows="dynamic", use_container_width=True)
            
            if st.button("🚀 生成 KMZ", type="primary"):
                kml, count = generate_kmz(final_df, coord_mode, cm)
                if count > 0:
                    kml.save("excel_import.kmz")
                    with open("excel_import.kmz", "rb") as f:
                        st.download_button("📥 下载文件", f, "excel_import.kmz")
                else:
                    st.error("生成失败。如果您选择了【度分秒】，请确保Excel里是 '度 分 秒' 的字符串格式。")
        except Exception as e:
            st.error(f"Excel 读取失败: {e}")

# ==========================================
# 模式 3：图片识别
# ==========================================
elif app_mode == "📸 图片识别":
    # 保持所有逻辑不变
    if 'angle' not in st.session_state: st.session_state.angle = 0.0
    if 'raw_img' not in st.session_state: st.session_state.raw_img = None
    if 'final_img' not in st.session_state: st.session_state.final_img = None
    if 'raw_text' not in st.session_state: st.session_state.raw_text = ""
    if 'parsed_df' not in st.session_state: st.session_state.parsed_df = None

    st.header("📸 图片识别")
    
    img_file = st.file_uploader("上传图片", type=['png', 'jpg', 'jpeg'])
    if img_file:
        if 'last_file_img' not in st.session_state or st.session_state.last_file_img != img_file.name:
            st.session_state.last_file_img = img_file.name
            st.session_state.raw_img = Image.open(img_file)
            st.session_state.angle = 0.0
            st.session_state.final_img = st.session_state.raw_img
            st.session_state.raw_text = ""
            st.session_state.parsed_df = None
            st.rerun()

    if st.session_state.raw_img:
        st.divider()
        st.subheader("1. 图像处理 (可选)")
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
        
        st.subheader("2. 识别设置")
        col1, col2 = st.columns([1, 1])
        with col1:
            coord_mode = st.selectbox("坐标格式", ["Decimal", "DMS", "DDM", "CGCS2000"], 
                                      format_func=lambda x: {
                                          "Decimal": "🔢 纯小数",
                                          "DMS": "🌐 度分秒",
                                          "DDM": "⏱️ 度+分",
                                          "CGCS2000": "📐 大地2000"
                                      }[x])
            cm = 0
            if coord_mode == "CGCS2000":
                cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
            st.write("---")
            thresh = st.slider("黑白阈值", 0, 255, 120)

        with col2:
            processed_preview = simple_preprocess(st.session_state.final_img, thresh)
            st.image(processed_preview, caption="最终识别图", use_column_width=True)

        if st.button("🔥 提取原始文本", type="primary", use_container_width=True):
            with st.spinner("提取中..."):
                final_processed = simple_preprocess(st.session_state.final_img, thresh)
                text = pytesseract.image_to_string(final_processed, lang='eng', config='--psm 6')
                st.session_state.raw_text = text
                st.session_state.parsed_df = None

        if st.session_state.raw_text:
            st.divider()
            st.subheader("3. 确认与编辑")
            edited_text = st.text_area("OCR结果 (可直接修改)", value=st.session_state.raw_text, height=200)
            
            if st.button("⚡ 解析表格数据", use_container_width=True):
                raw_data = extract_data_from_text(edited_text, coord_mode)
                if raw_data:
                    st.session_state.parsed_df = pd.DataFrame(raw_data)
                else:
                    st.error("无法解析数据")

        if st.session_state.parsed_df is not None:
            st.divider()
            st.subheader("4. 生成")
            final_df = st.data_editor(st.session_state.parsed_df, num_rows="dynamic", use_container_width=True)
            
            if st.button("🚀 生成 KMZ", type="primary"):
                kml, count = generate_kmz(final_df, coord_mode, cm)
                if count > 0:
                    kml.save("ocr_result.kmz")
                    with open("ocr_result.kmz", "rb") as f:
                        st.download_button("📥 下载文件", f, "ocr_result.kmz")
                else:
                    st.warning("无有效数据")
