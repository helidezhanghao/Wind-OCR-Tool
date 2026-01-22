import streamlit as st
import simplekml
import re
from pyproj import CRS, Transformer
import os
from PIL import Image
import pandas as pd
import numpy as np
from zhipuai import ZhipuAI
import json
import base64
from io import BytesIO
from streamlit_cropper import st_cropper

# --- 全局配置 ---
# 🔥 已更新为你提供的 Key (2026-01-22)
ZHIPU_API_KEY = "c1bcd3c427814b0b80e8edd72205a830.mWewm9ZI2UOgwYQy"

st.set_page_config(page_title="力力的坐标工具 v21.9", page_icon="🤖", layout="centered")

# ================= 工具函数 =================

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

def generate_kmz(df, coord_mode, cm=0):
    kml = simplekml.Kml()
    valid_count = 0
    for i, row in df.iterrows():
        try:
            # 兼容 AI 返回的字段名
            raw_v1 = row.get("纬度/X", row.get("Latitude", row.get("lat", 0)))
            raw_v2 = row.get("经度/Y", row.get("Longitude", row.get("lon", 0)))
            name = str(row.get("编号", row.get("ID", f"P{i+1}")))
            
            def clean_ai_val(val):
                if isinstance(val, (int, float)): return float(val)
                s_str = str(val).upper().replace('°', ' ').replace("'", ' ').replace('"', ' ').replace(':', ' ')
                parts = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", s_str)
                if len(parts) >= 3: return float(parts[0]) + float(parts[1])/60 + float(parts[2])/3600
                elif len(parts) >= 2: return float(parts[0]) + float(parts[1])/60
                elif len(parts) == 1: return float(parts[0])
                return 0.0

            v1 = clean_ai_val(raw_v1)
            v2 = clean_ai_val(raw_v2)
            
            lat, lon = 0, 0
            if coord_mode != "CGCS2000": lat, lon = (v1, v2) if v1 < v2 else (v2, v1)
            else:
                res, msg = to_wgs84(v1, v2, cm, False)
                if res: lat, lon = res, msg
                else: continue
            
            if abs(lat) > 0.1 and abs(lon) > 0.1:
                kml.newpoint(name=name, coords=[(lon, lat)])
                valid_count += 1
        except: continue
    return kml, valid_count

# --- 智谱 AI 识别核心函数 ---
def image_to_base64(image):
    """将 PIL 图片转换为带前缀的 Base64 字符串"""
    buffered = BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

def recognize_image_with_zhipu(image):
    """调用智谱 GLM-4V 进行视觉识别"""
    try:
        client = ZhipuAI(api_key=ZHIPU_API_KEY)
        img_base64 = image_to_base64(image)
        
        response = client.chat.completions.create(
            model="glm-4v",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请识别图片中的表格数据。直接提取 编号、纬度/X、经度/Y。请直接返回纯 JSON 数组字符串，不要用markdown代码块包裹。格式示例：[{\"编号\": \"T1\", \"纬度/X\": 34.12, \"经度/Y\": 115.33}]"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_base64
                            }
                        }
                    ]
                }
            ]
        )
        if not response.choices or not response.choices[0].message:
            return "Error: API 返回内容为空"
            
        return response.choices[0].message.content
    except Exception as e:
        return f"CRITICAL_ERROR: {str(e)}"

# ================= 界面主逻辑 =================

st.title("🤖 力力的坐标工具 v21.9")

# --- 侧边栏 ---
with st.sidebar:
    st.header("功能选择")
    app_mode = st.radio("请选择模式：", ["🖐️ 手动输入", "📊 Excel导入", "📸 AI图片识别"], index=2)
    st.divider()
    st.info("切换模式会清空当前数据")

# --- 模式 1: 手动输入 ---
if app_mode == "🖐️ 手动输入":
    st.header("🖐️ 手动录入坐标")
    c1, c2 = st.columns(2)
    with c1:
        coord_mode = st.selectbox("坐标格式", ["Decimal", "DMS", "DDM", "CGCS2000"])
    cm = 0
    with c2:
        if coord_mode == "CGCS2000":
            cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
            cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
    
    if 'manual_df' not in st.session_state:
        st.session_state.manual_df = pd.DataFrame([{"编号": "T1", "纬度/X": "", "经度/Y": ""}, {"编号": "T2", "纬度/X": "", "经度/Y": ""}])
    edited_df = st.data_editor(st.session_state.manual_df, num_rows="dynamic", use_container_width=True)
    if st.button("🚀 生成 KMZ", type="primary"):
        kml, count = generate_kmz(edited_df, coord_mode, cm)
        if count > 0:
            kml.save("manual.kmz")
            with open("manual.kmz", "rb") as f: st.download_button("📥 下载文件", f, "manual.kmz")
        else: st.error("数据无效")

# --- 模式 2: Excel导入 ---
elif app_mode == "📊 Excel导入":
    st.header("📊 Excel 批量导入")
    excel_file = st.file_uploader("上传 Excel", type=['xlsx', 'xls'])
    if excel_file:
        try:
            df = pd.read_excel(excel_file)
            st.success("读取成功")
            cols = list(df.columns)
            c1, c2, c3 = st.columns(3)
            with c1: col_name = st.selectbox("编号列", ["无"] + cols)
            with c2: col_lat = st.selectbox("纬度/X 列", cols, index=0)
            with c3: col_lon = st.selectbox("经度/Y 列", cols, index=0)
            
            processed = []
            for i, row in df.iterrows():
                processed.append({"编号": row[col_name] if col_name != "无" else f"P{i+1}", "纬度/X": row[col_lat], "经度/Y": row[col_lon]})
            proc_df = pd.DataFrame(processed)
            
            st.write("### 确认与生成")
            c_set1, c_set2 = st.columns(2)
            with c_set1: coord_mode = st.selectbox("坐标格式", ["Decimal", "DMS", "DDM", "CGCS2000"])
            cm = 0
            with c_set2:
                if coord_mode == "CGCS2000":
                    cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                    cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
            
            final_df = st.data_editor(proc_df, num_rows="dynamic", use_container_width=True)
            if st.button("🚀 生成 KMZ", type="primary"):
                kml, count = generate_kmz(final_df, coord_mode, cm)
                if count > 0:
                    kml.save("excel.kmz")
                    with open("excel.kmz", "rb") as f: st.download_button("📥 下载", f, "excel.kmz")
        except: st.error("读取失败")

# --- 模式 3: 智谱 AI 图片识别 ---
elif app_mode == "📸 AI图片识别":
    if 'raw_img' not in st.session_state: st.session_state.raw_img = None
    if 'ai_json_text' not in st.session_state: st.session_state.ai_json_text = ""
    if 'parsed_df' not in st.session_state: st.session_state.parsed_df = None

    st.header("📸 AI 视觉识别 (智谱GLM-4V)")
    
    img_file = st.file_uploader("上传图片", type=['png', 'jpg', 'jpeg'])
    
    if img_file:
        st.session_state.raw_img = Image.open(img_file)
        st.image(st.session_state.raw_img, caption="原始图片", use_column_width=True)
        
        if st.button("✨ 让智谱AI识别表格", type="primary"):
            with st.spinner("🚀 AI 正在努力识图中，请稍等..."):
                result = recognize_image_with_zhipu(st.session_state.raw_img)
            
            if result.startswith("CRITICAL_ERROR"):
                st.error("AI 接口调用失败！")
                st.error(result)
            elif result.startswith("Error"):
                st.warning(result)
            else:
                clean_result = result.replace("```json", "").replace("```", "").strip()
                st.session_state.ai_json_text = clean_result
                try:
                    data = json.loads(clean_result)
                    st.session_state.parsed_df = pd.DataFrame(data)
                    st.success("识别成功！")
                except:
                    st.error("AI 返回的数据格式有误，请在下方手动修正 JSON。")

    if st.session_state.ai_json_text:
        st.divider()
        st.subheader("📝 确认与编辑")
        with st.expander("查看 AI 原始返回"):
            st.text_area("JSON Raw", st.session_state.ai_json_text, height=100)

        if st.session_state.parsed_df is not None:
            st.caption("👇 请核对数据：")
            c1, c2 = st.columns(2)
            with c1:
                coord_mode = st.selectbox("图片里的坐标格式是？", ["Decimal", "DMS", "DDM", "CGCS2000"])
            cm = 0
            with c2:
                if coord_mode == "CGCS2000":
                    cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                    cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))

            final_df = st.data_editor(st.session_state.parsed_df, num_rows="dynamic", use_container_width=True)
            
            if st.button("🚀 生成 KMZ"):
                kml, count = generate_kmz(final_df, coord_mode, cm)
                if count > 0:
                    kml.save("zhipu_result.kmz")
                    with open("zhipu_result.kmz", "rb") as f:
                        st.download_button("📥 下载文件", f, "zhipu_result.kmz", type="primary")
                else:
                    st.error("无有效数据。")
