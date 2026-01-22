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

# --- 全局配置 (保持不变) ---
ZHIPU_API_KEY = "c1bcd3c427814b0b80e8edd72205a830.mWewm9ZI2UOgwYQy"
USER_PASSWORD = "2026"  # 用户密码
ADMIN_PASSWORD = "0521" # 管理员密码
LOG_FILE = "usage_log.csv"

# 设置 layout="wide"
st.set_page_config(page_title="力力的坐标工具 v24.1", page_icon="🎨", layout="wide")

# 🔥🔥🔥 全新 UI/CSS 设计 🔥🔥🔥
st.markdown("""
    <style>
        /* 全局字体和背景优化 */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8f9fa; /* 柔和灰背景 */
        }
        /* 移除顶部过多空白 */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
        }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;}

        /* --- 卡片式容器风格 --- */
        /* 给主要功能区添加白色卡片背景和阴影 */
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
             background-color: white;
             padding: 1.5rem;
             border-radius: 16px;
             box-shadow: 0 4px 20px rgba(0,0,0,0.06);
             margin-bottom: 1.5rem;
             border: 1px solid #f0f0f0;
        }

        /* --- 按钮美化 --- */
        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            height: 3.2em;
            font-weight: 600;
            font-size: 16px !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        /* 主按钮（生成/识别）强调色 */
        button[kind="primary"] {
            background-color: #007bff !important;
            border: none !important;
        }

        /* --- 登录界面专用 --- */
        .login-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 70vh;
        }
        .login-box {
            background: white;
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
            max-width: 450px;
            width: 100%;
        }
        .login-title { font-size: 1.8rem; font-weight: 700; color: #333; margin-bottom: 1rem; }
        .login-icon { font-size: 3rem; margin-bottom: 1rem; }

        /* --- 管理员卡片 --- */
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            text-align: center;
            border-bottom: 4px solid #007bff;
        }
        .metric-card h3 { color: #666; font-size: 1rem; font-weight: 600; margin-bottom: 5px;}
        .metric-card h1 { color: #333; font-size: 2.2rem; font-weight: 800; margin: 0;}
        .metric-card p { color: #888; font-size: 0.9rem; }
        
        /* --- 其他细节 --- */
        /* 调整标题样式 */
        h1, h2, h3 { color: #2c3e50; font-weight: 700 !important; }
        /* 分割线样式 */
        hr { margin: 2em 0; border-color: #eee; }
        /* 侧边栏背景 */
        [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #eee;}
    </style>
""", unsafe_allow_html=True)

# ================= 日志与工具函数 (完全不变) =================

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

# ================= 🚀 主程序逻辑 (UI重构) =================

if 'user_role' not in st.session_state:
    st.session_state.user_role = None

# --- 1. 登录界面 (美化版) ---
if st.session_state.user_role is None:
    # 使用 HTML wrapper 来应用 CSS 样式，让登录框居中且美观
    st.markdown("""
        <div class='login-wrapper'>
            <div class='login-box'>
                <div class='login-icon'>🔐</div>
                <div class='login-title'>力力坐标工具</div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        password = st.text_input("请输入访问密码", type="password")
        st.write("") # 空隙
        submit = st.form_submit_button("解锁进入", type="primary") # 使用主要按钮样式
        
        if submit:
            if password == USER_PASSWORD:
                st.session_state.user_role = 'user'
                log_event("Login", "User Access")
                st.toast("🎉 欢迎回来！正在进入系统...")
                st.rerun()
            elif password == ADMIN_PASSWORD:
                st.session_state.user_role = 'admin'
                st.toast("🛡️ 管理员模式已激活")
                st.rerun()
            else:
                st.error("密码错误，请重试")
    
    st.markdown("</div></div>", unsafe_allow_html=True) # 关闭 HTML wrapper

# --- 2. 管理员后台界面 (美化版) ---
elif st.session_state.user_role == 'admin':
    st.title("🛡️ 管理员后台监控")
    if st.sidebar.button("🔒 退出后台"):
        st.session_state.user_role = None
        st.rerun()

    df_logs = get_logs()
    total_visits = len(df_logs)
    ai_calls = len(df_logs[df_logs['Action'] == 'AI Recognize'])
    last_access = df_logs['Time'].iloc[-1] if not df_logs.empty else "无数据"

    # 使用新的卡片样式
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"<div class='metric-card'><h3>📊 总使用次数</h3><h1>{total_visits}</h1><p>累计操作记录</p></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card'><h3>📸 AI 识别次数</h3><h1>{ai_calls}</h1><p>调用大模型统计</p></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><h3>🕒 最近活动</h3><p style='font-size: 1.1rem; font-weight:bold; color:#333;'>{last_access}</p><p>最后操作时间</p></div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("📋 详细日志记录")
    # 使用 container 包裹表格，使其也有卡片效果
    with st.container():
        st.dataframe(df_logs.sort_index(ascending=False), use_container_width=True, height=400)
    st.download_button("📥 导出 CSV 日志", df_logs.to_csv(index=False).encode('utf-8'), "usage_logs.csv", "text/csv")


# --- 3. 普通用户界面 (美化版) ---
elif st.session_state.user_role == 'user':
    st.title("✨ 力力的坐标工具")
    
    with st.sidebar:
        st.markdown("### ⚙️ 控制台")
        if st.button("🔒 退出登录"):
            st.session_state.user_role = None
            st.rerun() 
        st.divider()
        app_mode = st.radio("选择功能模式：", ["🖐️ 手动输入", "📊 Excel导入", "📸 AI图片识别"], index=2)
        st.info("ℹ️ 切换模式将清空下方数据区域。")

    # 使用 st.container 创建卡片式布局
    with st.container():
        # 模式 1: 手动
        if app_mode == "🖐️ 手动输入":
            st.header("🖐️ 手动录入坐标")
            st.caption("请选择坐标格式并手动输入数据。")
            c1, c2 = st.columns(2)
            with c1: coord_mode = st.selectbox("1️⃣ 坐标格式", ["Decimal", "DMS", "DDM", "CGCS2000"])
            with c2:
                cm = 0
                if coord_mode == "CGCS2000":
                    cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                    cm = st.selectbox("2️⃣ 中央经线 (CGCS2000必选)", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
            
            st.divider()
            st.subheader("📝 数据编辑区域")
            if 'manual_df' not in st.session_state:
                st.session_state.manual_df = pd.DataFrame([{"编号": "T1", "纬度/X": "", "经度/Y": ""}, {"编号": "T2", "纬度/X": "", "经度/Y": ""}])
            edited_df = st.data_editor(st.session_state.manual_df, num_rows="dynamic", use_container_width=True)
            
            st.write("")
            if st.button("🚀 生成并下载 KMZ", type="primary"):
                log_event("Generate KMZ", "Manual")
                kml, count = generate_kmz(edited_df, coord_mode, cm)
                if count > 0:
                    kml.save("manual.kmz")
                    with open("manual.kmz", "rb") as f: st.download_button("📥 点击下载文件", f, "manual.kmz", type="primary")
                else: st.error("⚠️ 数据无效，请检查输入。")

        # 模式 2: Excel
        elif app_mode == "📊 Excel导入":
            st.header("📊 Excel 批量导入")
            st.caption("上传 Excel 文件并指定对应的列名。")
            excel_file = st.file_uploader("📄 点击上传 Excel 文件", type=['xlsx', 'xls'])
            if excel_file:
                try:
                    df = pd.read_excel(excel_file)
                    st.toast("✅ Excel 读取成功！")
                    st.divider()
                    st.subheader("🛠️ 列名映射配置")
                    cols = list(df.columns)
                    c1, c2, c3 = st.columns(3)
                    with c1: col_name = st.selectbox("编号列 (可选)", ["无"] + cols)
                    with c2: col_lat = st.selectbox("纬度/X 列 (必选)", cols, index=0)
                    with c3: col_lon = st.selectbox("经度/Y 列 (必选)", cols, index=0)
                    
                    processed = []
                    for i, row in df.iterrows():
                        processed.append({"编号": row[col_name] if col_name != "无" else f"P{i+1}", "纬度/X": row[col_lat], "经度/Y": row[col_lon]})
                    proc_df = pd.DataFrame(processed)
                    
                    st.divider()
                    st.subheader("📝 确认数据与格式")
                    c_set1, c_set2 = st.columns(2)
                    with c_set1: coord_mode = st.selectbox("1️⃣ 坐标格式", ["Decimal", "DMS", "DDM", "CGCS2000"])
                    with c_set2:
                        cm = 0
                        if coord_mode == "CGCS2000":
                            cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                            cm = st.selectbox("2️⃣ 中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
                    final_df = st.data_editor(proc_df, num_rows="dynamic", use_container_width=True)
                    
                    st.write("")
                    if st.button("🚀 生成并下载 KMZ", type="primary"):
                        log_event("Generate KMZ", "Excel")
                        kml, count = generate_kmz(final_df, coord_mode, cm)
                        if count > 0:
                            kml.save("excel.kmz")
                            with open("excel.kmz", "rb") as f: st.download_button("📥 点击下载文件", f, "excel.kmz", type="primary")
                except: st.error("❌ Excel 读取失败，请检查文件格式。")

        # 模式 3: AI
        elif app_mode == "📸 AI图片识别":
            st.header("📸 AI 视觉识别")
            st.caption("上传或拍摄包含坐标表格的图片，AI 将自动提取数据。")
            
            if 'raw_img' not in st.session_state: st.session_state.raw_img = None
            if 'ai_json_text' not in st.session_state: st.session_state.ai_json_text = ""
            if 'parsed_df' not in st.session_state: st.session_state.parsed_df = None
            
            img_file = st.file_uploader("🖼️ 图片上传 (点这里拍照或选图)", type=['png', 'jpg', 'jpeg'])
            
            if img_file:
                opened_img = Image.open(img_file)
                st.session_state.raw_img = ImageOps.exif_transpose(opened_img)
                st.image(st.session_state.raw_img, caption="已选图片预览", use_column_width=True)
                
                st.write("")
                if st.button("✨ 开始 AI 智能识别", type="primary"):
                    log_event("AI Recognize", "Start")
                    with st.spinner("🚀 AI 正在全力解读图片中，请稍候..."):
                        result = recognize_image_with_zhipu(st.session_state.raw_img)
                    if result.startswith("CRITICAL_ERROR"):
                        st.error(f"🤖 AI 接口调用失败: {result}")
                    elif result.startswith("Error"):
                        st.warning(f"🤖 AI 返回异常: {result}")
                    else:
                        clean_result = result.replace("```json", "").replace("```", "").strip()
                        st.session_state.ai_json_text = clean_result
                        try:
                            data = json.loads(clean_result)
                            st.session_state.parsed_df = pd.DataFrame(data)
                            st.toast("✅ 识别成功！请在下方核对数据。")
                        except: st.error("❌ AI 返回的数据格式无法解析，请重试。")

            if st.session_state.parsed_df is not None:
                st.divider()
                st.subheader("📝 结果核对与生成")
                st.caption("请务必确认下方的坐标格式选择与 AI 识别出的原始数据一致。")
                
                # 将设置和表格放在一个新的卡片容器中
                with st.container():
                    c1, c2 = st.columns(2)
                    with c1: coord_mode = st.selectbox("1️⃣ 图片中的坐标格式是？", ["Decimal (小数)", "DMS (度分秒)", "DDM (度.分)", "CGCS2000 (投影)"], index=0)
                    with c2:
                        cm = 0
                        if coord_mode == "CGCS2000 (投影)":
                            cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                            cm = st.selectbox("2️⃣ 中央经线 (CGCS2000必选)", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
                        else: st.empty()
                    
                    final_df = st.data_editor(st.session_state.parsed_df, num_rows="dynamic", use_container_width=True)
                    
                    st.write("")
                    if st.button("🚀 生成并下载 KMZ", type="primary"):
                        log_event("Generate KMZ", "AI Result")
                        mode_map = {"Decimal (小数)": "Decimal", "DMS (度分秒)": "DMS", "DDM (度.分)": "DDM", "CGCS2000 (投影)": "CGCS2000"}
                        kml, count = generate_kmz(final_df, mode_map[coord_mode], cm)
                        if count > 0:
                            kml.save("zhipu_result.kmz")
                            with open("zhipu_result.kmz", "rb") as f: st.download_button("📥 点击下载 KMZ 文件", f, "zhipu_result.kmz", type="primary")
                        else: st.error("⚠️ 无有效数据生成，请检查坐标格式选择。")
