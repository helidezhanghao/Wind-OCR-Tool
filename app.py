import streamlit as st
import simplekml
import re
from pyproj import CRS, Transformer
import os
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
from zhipuai import ZhipuAI
import json
import base64
from io import BytesIO, StringIO
from datetime import datetime
import csv
import zipfile
import xml.etree.ElementTree as ET

# --- 全局配置 ---
ZHIPU_API_KEY = "c1bcd3c427814b0b80e8edd72205a830.mWewm9ZI2UOgwYQy"
USER_PASSWORD = "2026"  # 普通用户密码
ADMIN_PASSWORD = "0521" # 管理员密码
LOG_FILE = "usage_log.csv"
LOGO_FILENAME = "logo.png"

# 设置 layout="wide"
st.set_page_config(page_title="力力的坐标工具 v31.2", page_icon="📲", layout="wide")

# 🔥🔥🔥 CSS 样式 (保持不变) 🔥🔥🔥
st.markdown("""
    <style>
        footer {display: none !important;}
        #MainMenu {display: none !important;}
        .stDeployButton {display: none !important;}
        
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
        }

        .login-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            height: auto;
            margin-bottom: 20px;
        }
        
        .login-box {
            background: #ffffff;
            padding: 0;
            border-radius: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            text-align: center;
            max-width: 400px;
            width: 100%;
            margin: auto;
            overflow: hidden;
            border: 1px solid #f0f0f0;
        }

        .login-banner-image {
            width: 100%;
            height: 200px;
            background-size: cover;
            background-position: center center !important; 
            background-repeat: no-repeat;
        }

        .login-content-wrapper {
            padding: 2rem 2.5rem 2.5rem 2.5rem;
        }
        
        .login-title { 
            font-size: 1.5rem; font-weight: 700; color: #333;
            margin-bottom: 1.5rem;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            height: 3em;
            font-weight: 600;
        }

        .metric-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #007bff;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-bottom: 10px;
        }
        
        @media (max-width: 768px) {
            [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 10px; }
            [data-testid="stHorizontalBlock"] > div { min-width: 100% !important; }
        }
    </style>
""", unsafe_allow_html=True)

# ================= 工具函数 =================

def get_local_image_base64(path):
    try:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded_string}" 
    except FileNotFoundError:
        return None

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Action", "Status"])

def log_event(action, status="Success"):
    init_log()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([current_time, action, status])

def get_logs():
    init_log()
    try: return pd.read_csv(LOG_FILE)
    except: return pd.DataFrame(columns=["Time", "Action", "Status"])

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
    # 智能列名匹配列表
    keys_v1 = ["纬度/X", "纬度", "Latitude", "lat", "Lat", "X", "x", "LAT", "Lat(N)"]
    keys_v2 = ["经度/Y", "经度", "Longitude", "lon", "Lon", "Y", "y", "LON", "Lon(E)"]
    keys_id = ["编号", "ID", "id", "Name", "name", "No"]

    for i, row in df.iterrows():
        try:
            raw_v1 = 0
            for k in keys_v1:
                if k in row:
                    raw_v1 = row[k]
                    break
            
            raw_v2 = 0
            for k in keys_v2:
                if k in row:
                    raw_v2 = row[k]
                    break
            
            name = f"P{i+1}"
            for k in keys_id:
                if k in row:
                    name = str(row[k])
                    break

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

# 万能格式解析器
def parse_universal_file(uploaded_file):
    fname = uploaded_file.name.lower()
    data_list = []
    
    # 1. 压缩包
    if fname.endswith(('.kmz', '.ovkmz', '.zip')):
        try:
            with zipfile.ZipFile(uploaded_file) as z:
                kml_files = [f for f in z.namelist() if f.lower().endswith(('.kml', '.ovkml'))]
                if kml_files:
                    with z.open(kml_files[0]) as f:
                        content = f.read().decode('utf-8', errors='ignore')
                        return parse_text_content(content, 'kml')
        except Exception as e:
            st.warning(f"压缩包解析失败: {e}")
            return None

    # 2. 文本类
    elif fname.endswith(('.kml', '.ovkml', '.gpx', '.plt')):
        content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        return parse_text_content(content, fname.split('.')[-1])

    # 3. DXF
    elif fname.endswith('.dxf'):
        content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        return parse_dxf_regex(content)

    # 4. SHP
    elif fname.endswith('.shp'):
        try:
            import geopandas as gpd
            with st.spinner("正在尝试通过 Geopandas 读取 SHP..."):
                with open("temp.shp", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                gdf = gpd.read_file("temp.shp")
                if gdf.crs:
                    gdf = gdf.to_crs(epsg=4326)
                
                for idx, row in gdf.iterrows():
                    geom = row.geometry
                    if geom.geom_type == 'Point':
                        data_list.append({"编号": f"P{idx}", "纬度": geom.y, "经度": geom.x})
                    else:
                        data_list.append({"编号": f"P{idx}", "纬度": geom.centroid.y, "经度": geom.centroid.x})
            return pd.DataFrame(data_list)
        except ImportError:
            st.error("服务器未安装 Geopandas，无法解析 SHP。")
            return None
        except Exception as e:
            st.error(f"SHP 解析失败: {e}")
            return None

    # 5. 加密/二进制 (OVBJ, OVOBJ, DWG)
    else:
        st.info(f"⚠️ 检测到二进制/加密格式 ({fname})，正在尝试暴力提取坐标...")
        try:
            content = uploaded_file.getvalue().decode('latin-1', errors='ignore') 
            return parse_regex_brute_force(content)
        except:
            return None

    return None

def parse_text_content(content, fmt):
    data = []
    if 'kml' in fmt:
        try:
            content = re.sub(r'xmlns="[^"]+"', '', content, count=1)
            root = ET.fromstring(content)
            for placemark in root.findall(".//Placemark"):
                name = placemark.find("name")
                name_txt = name.text if name is not None else "NoName"
                coords = placemark.find(".//coordinates")
                if coords is not None and coords.text:
                    c_str = coords.text.strip().split()[0] 
                    parts = c_str.split(',')
                    if len(parts) >= 2:
                        data.append({"编号": name_txt, "纬度": float(parts[1]), "经度": float(parts[0])})
        except: pass

    elif 'gpx' in fmt:
        try:
            root = ET.fromstring(content)
            for wpt in root.findall(".//wpt"):
                lat = wpt.get("lat")
                lon = wpt.get("lon")
                name = wpt.find("name")
                name_txt = name.text if name is not None else "WPT"
                if lat and lon:
                    data.append({"编号": name_txt, "纬度": float(lat), "经度": float(lon)})
        except: pass

    elif 'plt' in fmt:
        lines = content.splitlines()
        for line in lines[6:]: 
            parts = line.strip().split(',')
            if len(parts) > 4:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    data.append({"编号": "PLT_PT", "纬度": lat, "经度": lon})
                except: pass
    
    return pd.DataFrame(data)

def parse_dxf_regex(content):
    data = []
    lines = content.splitlines()
    x, y = None, None
    for i, line in enumerate(lines):
        line = line.strip()
        if line == '10': # X
            try: x = float(lines[i+1].strip())
            except: pass
        if line == '20': # Y
            try: y = float(lines[i+1].strip())
            except: pass
        
        if x is not None and y is not None:
            data.append({"编号": "DXF_PT", "纬度": y, "经度": x})
            x, y = None, None 
    return pd.DataFrame(data)

def parse_regex_brute_force(content):
    data = []
    pattern = r"(\d{2,3}\.\d{4,}),?\s?(\d{2,3}\.\d{4,})"
    matches = re.findall(pattern, content)
    for i, (v1, v2) in enumerate(matches):
        try:
            val1, val2 = float(v1), float(v2)
            lat, lon = (val2, val1) if val1 > val2 else (val1, val2)
            if 0 < lat < 90 and 0 < lon < 180:
                data.append({"编号": f"RAW_{i}", "纬度": lat, "经度": lon})
        except: pass
    return pd.DataFrame(data)

def image_to_base64(image):
    buffered = BytesIO()
    if image.mode != "RGB": image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

def recognize_image_with_zhipu(image):
    try:
        client = ZhipuAI(api_key=ZHIPU_API_KEY)
        img_base64 = image_to_base64(image)
        response = client.chat.completions.create(
            model="glm-4v-flash",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请识别图片中的表格数据。直接提取 编号、纬度/X、经度/Y。请直接返回纯 JSON 数组字符串。⚠️ 重要原则：**所见即所得**。如果图片里的数字是小数，请直接返回小数。如果图片里的数字是度分秒，请保持原样返回字符串。不要进行任何格式转换。"},
                    {"type": "image_url", "image_url": {"url": img_base64}}
                ]
            }]
        )
        if not response.choices or not response.choices[0].message: return "Error: API 返回内容为空"
        return response.choices[0].message.content
    except Exception as e: return f"CRITICAL_ERROR: {str(e)}"

# ================= 🚀 主程序逻辑 =================

if 'user_role' not in st.session_state:
    st.session_state.user_role = None 
if 'login_mode' not in st.session_state:
    st.session_state.login_mode = 'select'

# --- 1. 登录界面 ---
if st.session_state.user_role is None:
    logo_b64 = get_local_image_base64(LOGO_FILENAME)
    bg_style = f"background-image: url('{logo_b64}');" if logo_b64 else "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);"

    c_left, c_center, c_right = st.columns([2, 1, 2])
    with c_center:
        st.markdown(f"""
            <div class='login-wrapper'>
                <div class='login-box'>
                    <div class='login-banner-image' style="{bg_style}"></div>
                    <div class='login-content-wrapper'>
                        <div class='login-title'>力力坐标工具</div>
        """, unsafe_allow_html=True)
        
        if st.session_state.login_mode == 'select':
            b_gap1, b_content, b_gap2 = st.columns([1, 3, 1])
            with b_content:
                if st.button("🚀 普通用户登录", type="primary", use_container_width=True):
                    st.session_state.login_mode = 'user_input'
                    st.rerun()
                st.write("")
                if st.button("🛡️ 管理员登录", use_container_width=True):
                    st.session_state.login_mode = 'admin_input'
                    st.rerun()

        elif st.session_state.login_mode == 'user_input':
            st.caption("🔒 请输入普通用户密码")
            with st.form("user_login_form"):
                password = st.text_input("用户密码", type="password", label_visibility="collapsed")
                submit = st.form_submit_button("解锁进入", type="primary", use_container_width=True)
                if submit:
                    if password == USER_PASSWORD:
                        st.session_state.user_role = 'user'
                        st.session_state.login_mode = 'select'
                        log_event("Login", "User Access")
                        st.toast("欢迎回来！")
                        st.rerun()
                    else: st.error("密码错误")
            b_gap1, b_back, b_gap2 = st.columns([1, 3, 1])
            with b_back:
                if st.button("⬅️ 返回", use_container_width=True):
                    st.session_state.login_mode = 'select'
                    st.rerun()

        elif st.session_state.login_mode == 'admin_input':
            st.caption("🔒 请输入管理员密码")
            with st.form("admin_login_form"):
                password = st.text_input("管理员密码", type="password", label_visibility="collapsed")
                submit = st.form_submit_button("解锁后台", type="primary", use_container_width=True)
                if submit:
                    if password == ADMIN_PASSWORD:
                        st.session_state.user_role = 'admin'
                        st.session_state.login_mode = 'select'
                        st.toast("管理员身份已验证")
                        st.rerun()
                    else: st.error("密码错误")
            b_gap1, b_back, b_gap2 = st.columns([1, 3, 1])
            with b_back:
                if st.button("⬅️ 返回", use_container_width=True):
                    st.session_state.login_mode = 'select'
                    st.rerun()

        st.markdown("</div></div></div>", unsafe_allow_html=True)

# --- 2. 管理员后台界面 ---
elif st.session_state.user_role == 'admin':
    st.title("🛡️ 管理员后台")
    if st.sidebar.button("🔒 退出"):
        st.session_state.user_role = None
        st.rerun()

    df_logs = get_logs()
    total_visits = len(df_logs)
    ai_calls = len(df_logs[df_logs['Action'] == 'AI Recognize'])
    last_access = df_logs['Time'].iloc[-1] if not df_logs.empty else "无数据"

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"<div class='metric-card'><h3>📊 总使用次数</h3><h1>{total_visits}</h1></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card'><h3>📸 AI 调用</h3><h1>{ai_calls}</h1></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><h3>🕒 最近活动</h3><p>{last_access}</p></div>", unsafe_allow_html=True)

    st.subheader("详细日志")
    st.dataframe(df_logs.sort_index(ascending=False), use_container_width=True)
    st.download_button("📥 导出 CSV", df_logs.to_csv(index=False).encode('utf-8'), "usage_logs.csv", "text/csv")


# --- 3. 普通用户界面 ---
elif st.session_state.user_role == 'user':
    
    with st.sidebar:
        if st.button("🔒 退出登录"):
            st.session_state.user_role = None
            st.rerun() 
        st.divider()
        app_mode = st.radio("功能选择", ["🖐️ 手动输入", "📄 文本导入", "🛠️ 万能格式转换", "📸 AI图片识别"], index=2)
        st.info("切换模式会清空当前数据")

    st.title("力力的坐标工具 v31.2")
    
    # 模式 1: 手动
    if app_mode == "🖐️ 手动输入":
        st.header("🖐️ 手动录入")
        c1, c2 = st.columns(2)
        with c1: coord_mode = st.selectbox("坐标格式", ["Decimal", "DMS", "DDM", "CGCS2000"])
        with c2:
            cm = 0
            if coord_mode == "CGCS2000":
                cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
        if 'manual_df' not in st.session_state:
            st.session_state.manual_df = pd.DataFrame([{"编号": "T1", "纬度/X": "", "经度/Y": ""}, {"编号": "T2", "纬度/X": "", "经度/Y": ""}])
        edited_df = st.data_editor(st.session_state.manual_df, num_rows="dynamic", use_container_width=True)
        
        if st.button("🚀 生成 KMZ", type="primary"):
            log_event("Generate KMZ", "Manual")
            kml, count = generate_kmz(edited_df, coord_mode, cm)
            if count > 0:
                kml.save("manual.kmz")
                with open("manual.kmz", "rb") as f: st.download_button("📥 下载文件", f, "manual.kmz", type="primary")
            else: st.error("数据无效")

    # 模式 2: 文本导入 (拖拽增强)
    elif app_mode == "📄 文本导入":
        st.header("📄 文本导入 (Excel/TXT/CSV)")
        # 🔥🔥🔥 提示语增强 🔥🔥🔥
        file_buffer = st.file_uploader("📄 点击上传或直接拖拽文件到此处 (Excel/TXT/CSV)", type=['xlsx', 'xls', 'csv', 'txt'])
        if file_buffer:
            try:
                fname = file_buffer.name.lower()
                if fname.endswith(('.csv', '.txt')):
                    df = pd.read_csv(file_buffer, sep=None, engine='python')
                else:
                    df = pd.read_excel(file_buffer)
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
                
                st.write("---")
                c_set1, c_set2 = st.columns(2)
                with c_set1: coord_mode = st.selectbox("坐标格式", ["Decimal", "DMS", "DDM", "CGCS2000"])
                with c_set2:
                    cm = 0
                    if coord_mode == "CGCS2000":
                        cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                        cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
                final_df = st.data_editor(proc_df, num_rows="dynamic", use_container_width=True)
                
                if st.button("🚀 生成 KMZ", type="primary"):
                    log_event("Generate KMZ", "Text Import")
                    kml, count = generate_kmz(final_df, coord_mode, cm)
                    if count > 0:
                        kml.save("text_import.kmz")
                        with open("text_import.kmz", "rb") as f: st.download_button("📥 下载文件", f, "text_import.kmz", type="primary")
            except Exception as e: st.error(f"读取失败: {str(e)}")

    # 模式 3: 万能格式转换
    elif app_mode == "🛠️ 万能格式转换":
        st.header("🛠️ 万能格式转换 (硬解一切)")
        st.caption("支持格式: kml, kmz, ovkml, ovkmz, gpx, plt, dxf, shp, ovbj, ovobj")
        
        # 🔥🔥🔥 提示语增强 🔥🔥🔥
        uni_file = st.file_uploader("📂 点击上传或直接拖拽格式文件到此处", type=['kml', 'kmz', 'ovkml', 'ovkmz', 'gpx', 'plt', 'dxf', 'dwg', 'shp', 'ovbj', 'ovobj'])
        
        if uni_file:
            with st.spinner("🚀 正在暴力解析文件..."):
                parsed_df = parse_universal_file(uni_file)
            
            if parsed_df is not None and not parsed_df.empty:
                st.success(f"✅ 解析成功！共找到 {len(parsed_df)} 个点")
                st.dataframe(parsed_df)
                
                if st.button("🚀 立即转换为 KMZ", type="primary"):
                    log_event("Generate KMZ", "Universal Hard Decode")
                    kml, count = generate_kmz(parsed_df, "Decimal", 0)
                    if count > 0:
                        kml.save("universal_output.kmz")
                        with open("universal_output.kmz", "rb") as f: 
                            st.download_button("📥 下载 KMZ", f, "universal_output.kmz", type="primary")
                    else: st.error("生成失败")
            else:
                st.error("❌ 解析失败或文件中没有提取到有效坐标。")

    # 模式 4: AI (拖拽增强)
    elif app_mode == "📸 AI图片识别":
        st.header("📸 AI 识别")
        if 'raw_img' not in st.session_state: st.session_state.raw_img = None
        if 'ai_json_text' not in st.session_state: st.session_state.ai_json_text = ""
        if 'parsed_df' not in st.session_state: st.session_state.parsed_df = None
        
        # 🔥🔥🔥 提示语增强 🔥🔥🔥
        img_file = st.file_uploader("📸 点击上传或直接拖拽图片到此处 (拍照/选图)", type=['png', 'jpg', 'jpeg'])
        if img_file:
            opened_img = Image.open(img_file)
            st.session_state.raw_img = ImageOps.exif_transpose(opened_img)
            st.image(st.session_state.raw_img, caption="预览", width=350)
            
            if st.button("✨ 开始识别", type="primary"):
                log_event("AI Recognize", "Start")
                with st.spinner("AI 识别中..."):
                    result = recognize_image_with_zhipu(st.session_state.raw_img)
                if result.startswith("CRITICAL_ERROR"):
                    st.error(f"失败: {result}")
                elif result.startswith("Error"):
                    st.warning(result)
                else:
                    clean_result = result.replace("```json", "").replace("```", "").strip()
                    st.session_state.ai_json_text = clean_result
                    try:
                        data = json.loads(clean_result)
                        st.session_state.parsed_df = pd.DataFrame(data)
                        st.success("识别成功！")
                    except: st.error("格式解析错误")

        if st.session_state.parsed_df is not None:
            st.divider()
            st.subheader("结果核对")
            c1, c2 = st.columns(2)
            with c1: coord_mode = st.selectbox("图片坐标格式", ["Decimal (小数)", "DMS (度分秒)", "DDM (度.分)", "CGCS2000 (投影)"], index=0)
            with c2:
                cm = 0
                if coord_mode == "CGCS2000 (投影)":
                    cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                    cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
            
            final_df = st.data_editor(st.session_state.parsed_df, num_rows="dynamic", use_container_width=True)
            
            st.write("")
            if st.button("🚀 生成 KMZ", type="primary"):
                log_event("Generate KMZ", "AI Result")
                mode_map = {"Decimal (小数)": "Decimal", "DMS (度分秒)": "DMS", "DDM (度.分)": "DDM", "CGCS2000 (投影)": "CGCS2000"}
                kml, count = generate_kmz(final_df, mode_map[coord_mode], cm)
                if count > 0:
                    kml.save("zhipu_result.kmz")
                    with open("zhipu_result.kmz", "rb") as f: st.download_button("📥 下载文件", f, "zhipu_result.kmz", type="primary")
                else: st.error("无有效数据。")
