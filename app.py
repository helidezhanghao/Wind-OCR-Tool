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
from io import BytesIO
from datetime import datetime
import csv

# --- 全局配置 ---
ZHIPU_API_KEY = "c1bcd3c427814b0b80e8edd72205a830.mWewm9ZI2UOgwYQy"

# 🔑 双密码配置
USER_PASSWORD = "2026"  # 用户密码
ADMIN_PASSWORD = "0521" # 管理员密码

# 📊 日志文件路径
LOG_FILE = "usage_log.csv"

# 设置 layout="wide" 让手机端尽量撑满
st.set_page_config(page_title="力力的坐标工具 v24.0", page_icon="🔐", layout="wide")

# 🔥🔥🔥 CSS 样式注入 🔥🔥🔥
st.markdown("""
    <style>
        .block-container {padding-top: 1rem !important; padding-left: 1rem !important; padding-right: 1rem !important;}
        #MainMenu {visibility: hidden;} footer {visibility: hidden;}
        div.stButton > button {width: 100%; border-radius: 8px; height: 3em; font-weight: bold; font-size: 16px !important;}
        .login-container {padding: 30px; border-radius: 10px; background-color: #f0f2f6; margin-top: 50px; text-align: center;}
        /* 管理员界面的卡片样式 */
        .metric-card {background-color: #e8f4f8; padding: 20px; border-radius: 10px; border-left: 5px solid #00a2ff; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# ================= 📊 日志系统函数 =================

def init_log():
    """初始化日志文件"""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Action", "Status"])

def log_event(action, status="Success"):
    """记录一条使用日志"""
    init_log()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([current_time, action, status])

def get_logs():
    """读取所有日志"""
    init_log()
    try:
        return pd.read_csv(LOG_FILE)
    except:
        return pd.DataFrame(columns=["Time", "Action", "Status"])

# ================= 🛠️ 核心工具函数 =================

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

def image_to_base64(image):
    buffered = BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

def recognize_image_with_zhipu(image):
    try:
        client = ZhipuAI(api_key=ZHIPU_API_KEY)
        img_base64 = image_to_base64(image)
        response = client.chat.completions.create(
            model="glm-4v-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请识别图片中的表格数据。直接提取 编号、纬度/X、经度/Y。请直接返回纯 JSON 数组字符串。⚠️ 重要原则：**所见即所得**。如果图片里的数字是小数，请直接返回小数。如果图片里的数字是度分秒，请保持原样返回字符串。不要进行任何格式转换。"
                        },
                        {"type": "image_url", "image_url": {"url": img_base64}}
                    ]
                }
            ]
        )
        if not response.choices or not response.choices[0].message:
            return "Error: API 返回内容为空"
        return response.choices[0].message.content
    except Exception as e:
        return f"CRITICAL_ERROR: {str(e)}"

# ================= 🚀 主程序逻辑 =================

if 'user_role' not in st.session_state:
    st.session_state.user_role = None # None, 'user', 'admin'

# --- 1. 登录界面 ---
if st.session_state.user_role is None:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("🔐 力力坐标工具")
    
    with st.form("login_form"):
        st.info("请输入访问密码")
        password = st.text_input("密码", type="password", label_visibility="collapsed")
        submit = st.form_submit_button("解锁进入")
        
        if submit:
            if password == USER_PASSWORD:
                st.session_state.user_role = 'user'
                log_event("Login", "User Access") # 记录登录
                st.success("欢迎回来！")
                st.rerun()
            elif password == ADMIN_PASSWORD:
                st.session_state.user_role = 'admin'
                # log_event("Login", "Admin Access") # 管理员登录不一定非要记
                st.success("管理员模式已激活")
                st.rerun()
            else:
                st.error("密码错误")

# --- 2. 管理员后台界面 (0521) ---
elif st.session_state.user_role == 'admin':
    st.title("🛡️ 管理员后台监控")
    
    if st.sidebar.button("🔒 退出后台"):
        st.session_state.user_role = None
        st.rerun()

    # 获取数据
    df_logs = get_logs()
    
    # 统计数据
    total_visits = len(df_logs)
    ai_calls = len(df_logs[df_logs['Action'] == 'AI Recognize'])
    kmz_gens = len(df_logs[df_logs['Action'] == 'Generate KMZ'])
    last_access = df_logs['Time'].iloc[-1] if not df_logs.empty else "无数据"

    # 展示卡片
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='metric-card'><h3>📊 总使用次数</h3><h1>{total_visits}</h1></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><h3>📸 AI 识别次数</h3><h1>{ai_calls}</h1></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><h3>🕒 最后访问</h3><p>{last_access}</p></div>", unsafe_allow_html=True)

    st.subheader("📋 详细使用记录")
    # 显示最新的记录在最上面
    st.dataframe(df_logs.sort_index(ascending=False), use_container_width=True)

    st.download_button("📥 导出日志 CSV", df_logs.to_csv(index=False).encode('utf-8'), "usage_logs.csv", "text/csv")


# --- 3. 普通用户界面 (2026) ---
elif st.session_state.user_role == 'user':
    st.title("📸 力力的坐标工具 v24.0")
    
    with st.sidebar:
        if st.button("🔒 退出登录"):
            st.session_state.user_role = None
            st.rerun() 
        st.header("功能选择")
        app_mode = st.radio("请选择模式：", ["🖐️ 手动输入", "📊 Excel导入", "📸 AI图片识别"], index=2)
        st.divider()
        st.info("切换模式会清空当前数据")

    # 模式 1: 手动
    if app_mode == "🖐️ 手动输入":
        st.header("🖐️ 手动录入坐标")
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
            log_event("Generate KMZ", "Manual") # 📝 记账
            kml, count = generate_kmz(edited_df, coord_mode, cm)
            if count > 0:
                kml.save("manual.kmz")
                with open("manual.kmz", "rb") as f: st.download_button("📥 下载文件", f, "manual.kmz")
            else: st.error("数据无效")

    # 模式 2: Excel
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
                with c_set2:
                    cm = 0
                    if coord_mode == "CGCS2000":
                        cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                        cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
                final_df = st.data_editor(proc_df, num_rows="dynamic", use_container_width=True)
                if st.button("🚀 生成 KMZ", type="primary"):
                    log_event("Generate KMZ", "Excel") # 📝 记账
                    kml, count = generate_kmz(final_df, coord_mode, cm)
                    if count > 0:
                        kml.save("excel.kmz")
                        with open("excel.kmz", "rb") as f: st.download_button("📥 下载", f, "excel.kmz")
            except: st.error("读取失败")

    # 模式 3: AI
    elif app_mode == "📸 AI图片识别":
        if 'raw_img' not in st.session_state: st.session_state.raw_img = None
        if 'ai_json_text' not in st.session_state: st.session_state.ai_json_text = ""
        if 'parsed_df' not in st.session_state: st.session_state.parsed_df = None
        
        img_file = st.file_uploader("📸 图片上传 (点这里拍照或选图)", type=['png', 'jpg', 'jpeg'])
        
        if img_file:
            opened_img = Image.open(img_file)
            st.session_state.raw_img = ImageOps.exif_transpose(opened_img)
            st.image(st.session_state.raw_img, caption="预览", use_column_width=True)
            
            if st.button("✨ 开始 AI 识别", type="primary"):
                log_event("AI Recognize", "Start") # 📝 记账
                with st.spinner("🚀 AI 正在努力识图中..."):
                    result = recognize_image_with_zhipu(st.session_state.raw_img)
                if result.startswith("CRITICAL_ERROR"):
                    st.error("AI 接口调用失败！")
                elif result.startswith("Error"):
                    st.warning(result)
                else:
                    clean_result = result.replace("```json", "").replace("```", "").strip()
                    st.session_state.ai_json_text = clean_result
                    try:
                        data = json.loads(clean_result)
                        st.session_state.parsed_df = pd.DataFrame(data)
                        st.success("识别成功！")
                    except: st.error("AI 返回的数据格式有误")

        if st.session_state.ai_json_text:
            st.divider()
            st.subheader("📝 结果核对")
            if st.session_state.parsed_df is not None:
                c1, c2 = st.columns(2)
                with c1: coord_mode = st.selectbox("坐标格式", ["Decimal (小数)", "DMS (度分秒)", "DDM (度.分)", "CGCS2000 (投影)"], index=0)
                with c2:
                    cm = 0
                    if coord_mode == "CGCS2000 (投影)":
                        cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                        cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
                final_df = st.data_editor(st.session_state.parsed_df, num_rows="dynamic", use_container_width=True)
                st.write("")
                if st.button("🚀 生成 KMZ 文件"):
                    log_event("Generate KMZ", "AI Result") # 📝 记账
                    mode_map = {"Decimal (小数)": "Decimal", "DMS (度分秒)": "DMS", "DDM (度.分)": "DDM", "CGCS2000 (投影)": "CGCS2000"}
                    kml, count = generate_kmz(final_df, mode_map[coord_mode], cm)
                    if count > 0:
                        kml.save("zhipu_result.kmz")
                        with open("zhipu_result.kmz", "rb") as f: st.download_button("📥 点击下载 KMZ", f, "zhipu_result.kmz", type="primary")
                    else: st.error("无有效数据。")
