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
st.set_page_config(page_title="力力的坐标工具 v26.1 (Mobile Tweaks)", page_icon="📲", layout="wide")

# 🔥🔥🔥 核心：深度定制 CSS 以实现逼真的 iOS 风格 🔥🔥🔥
st.markdown("""
    <style>
        /* --- 定义 iOS 颜色变量 --- */
        :root {
            --ios-bg: #F2F2F7;             /* 系统背景灰 */
            --ios-card-bg: #FFFFFF;        /* 卡片纯白 */
            --ios-blue: #007AFF;           /* 官方蓝色 */
            --ios-text-primary: #000000;   /* 主要文本 */
            --ios-text-secondary: #8E8E93; /* 次要文本 */
            --ios-input-bg: #EBEBF0;       /* 输入框填充灰 */
            --ios-divider: #C6C6C8;        /* 分割线 */
        }

        /* --- 1. 全局设置 --- */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
            background-color: var(--ios-bg) !important;
            color: var(--ios-text-primary);
        }
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 4rem !important;
            max-width: 960px;
        }
        footer {display: none !important;}
        #MainMenu {display: none !important;}
        .stDeployButton {display: none !important;}

        /* --- 2. 标题与文本 --- */
        h1 { font-weight: 800 !important; font-size: 2rem !important; letter-spacing: -0.5px; margin-bottom: 1rem !important; }
        h2, h3 { font-weight: 700 !important; color: #1C1C1E; }
        .stCaption, p small { color: var(--ios-text-secondary) !important; font-size: 0.95rem; }
        hr { border-color: var(--ios-divider); opacity: 0.5; margin: 1.5em 0; }

        /* --- 3. iOS 风格卡片容器 --- */
        [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] {
             background-color: var(--ios-card-bg);
             border-radius: 20px;
             padding: 24px;
             box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.03);
             margin-bottom: 24px; border: none;
        }
        [data-testid="stSidebar"] { background-color: var(--ios-card-bg); border-right: 1px solid #E5E5EA; }

        /* --- 4. iOS 控件风格 --- */
        [data-testid="stTextInput"] input {
            background-color: var(--ios-input-bg) !important; border: none !important;
            border-radius: 12px !important; height: 48px; padding: 0 16px; font-size: 17px;
        }
        [data-testid="stSelectbox"] div[class*="control"] {
            background-color: var(--ios-input-bg) !important; border: none !important;
            border-radius: 12px !important; height: 48px;
        }
        [data-testid='stFileUploader'] section {
            border-radius: 16px; background-color: var(--ios-input-bg); border: 2px dashed #D1D1D6;
        }

        /* --- 5. 按钮美化 --- */
        div.stButton > button {
            border-radius: 100px !important; height: 52px; font-weight: 600;
            font-size: 17px !important; border: none !important; box-shadow: none !important;
            background-color: #E5E5EA; color: var(--ios-blue) !important;
            transition: transform 0.15s ease;
        }
        div.stButton > button:active { transform: scale(0.97); background-color: #D1D1D6; }
        button[kind="primary"] { background-color: var(--ios-blue) !important; color: white !important; }

        /* --- 6. 登录界面专用样式 (微调优化) --- */
        .login-wrapper { display: flex; justify-content: center; align-items: center; min-height: 70vh; }
        .login-box {
            background: var(--ios-card-bg);
            /* padding: 3rem 2.5rem; <-- 旧值，太大 */
            padding: 2.5rem 2rem; /* <-- 新值：减小内边距 */
            border-radius: 32px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.06);
            text-align: center;
            /* max-width: 420px; width: 90%; <-- 旧值 */
            max-width: 400px; width: 94%; /* <-- 新值：在小屏上占更多宽度 */
            margin: auto;
        }
        .login-icon { font-size: 4.5rem; margin-bottom: 0.5rem; }
        .login-title { font-size: 1.8rem; font-weight: 800; margin-bottom: 2rem; color: #000;}

        /* --- 7. 管理员卡片 --- */
        .metric-card {
            background-color: var(--ios-card-bg); padding: 24px; border-radius: 22px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.03); text-align: center;
            height: 100%; /* 确保高度一致 */
        }
        .metric-card h3 { font-size: 0.85rem; color: var(--ios-text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; }
        .metric-card h1 { font-size: 2.5rem; font-weight: 800; color: var(--ios-text-primary); margin: 0; line-height: 1.1;}
        
        /* --- 8. 移动端适配微调 (关键修改) --- */
        /* 在小屏幕上强制 Admin 卡片堆叠显示，而不是挤成三列 */
        @media (max-width: 768px) {
            /* 找到包裹三个卡片的容器，强制它换行 */
            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
                gap: 16px; /* 卡片间距 */
            }
            /* 让每个卡片占满一行 */
            [data-testid="stHorizontalBlock"] > div {
                min-width: 100% !important;
                flex: 1 1 auto !important;
                margin-bottom: 8px;
            }
        }
        /* 调整登录框在超小屏上的内边距 */
        @media (max-width: 380px) {
            .login-box { padding: 2rem 1.5rem; }
            .login-title { font-size: 1.6rem; }
        }
        
        img { border-radius: 16px; }
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

# ================= 🚀 主程序逻辑 (iOS 风格重构) =================

if 'user_role' not in st.session_state:
    st.session_state.user_role = None

# --- 1. 登录界面 (iOS弹窗风格，微调) ---
if st.session_state.user_role is None:
    st.markdown("""
        <div class='login-wrapper'>
            <div class='login-box'>
                <div class='login-icon'>📲</div>
                <div class='login-title'>力力坐标工具</div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        password = st.text_input("密码", type="password", placeholder="请输入访问密码", label_visibility="collapsed")
        st.write("")
        submit = st.form_submit_button("解锁进入", type="primary")
        
        if submit:
            if password == USER_PASSWORD:
                st.session_state.user_role = 'user'
                log_event("Login", "User Access")
                st.toast("🎉 验证成功")
                st.rerun()
            elif password == ADMIN_PASSWORD:
                st.session_state.user_role = 'admin'
                st.toast("🛡️ 管理员模式")
                st.rerun()
            else:
                st.error("密码错误")
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# --- 2. 管理员后台界面 (iOS Widget风格，微调) ---
elif st.session_state.user_role == 'admin':
    st.title("管理员控制台")
    
    with st.container():
        c_btn, c_title = st.columns([1, 5])
        with c_btn:
             if st.button("🔒 退出"):
                st.session_state.user_role = None
                st.rerun()

        df_logs = get_logs()
        total_visits = len(df_logs)
        ai_calls = len(df_logs[df_logs['Action'] == 'AI Recognize'])
        last_access = df_logs['Time'].iloc[-1] if not df_logs.empty else "无数据"

        # iOS Widget 风格卡片 (手机端会自动堆叠)
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f"<div class='metric-card'><h3>总使用量</h3><h1>{total_visits}</h1></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='metric-card'><h3>AI 调用</h3><h1>{ai_calls}</h1></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='metric-card'><h3>最近活动</h3><p style='font-size: 1rem; font-weight:600; color:#1C1C1E; margin-top:10px;'>{last_access}</p></div>", unsafe_allow_html=True)

    st.subheader("访问日志")
    with st.container():
        st.dataframe(df_logs.sort_index(ascending=False), use_container_width=True, height=400)
        st.write("")
        st.download_button("📥 导出日志记录", df_logs.to_csv(index=False).encode('utf-8'), "usage_logs.csv", "text/csv")


# --- 3. 普通用户界面 (iOS App风格) ---
elif st.session_state.user_role == 'user':
    
    with st.sidebar:
        st.markdown("### 设置")
        if st.button("🔒 退出登录"):
            st.session_state.user_role = None
            st.rerun() 
        st.divider()
        st.markdown("### 模式选择")
        app_mode = st.radio("模式选择", ["🖐️ 手动输入", "📊 Excel导入", "📸 AI图片识别"], index=2, label_visibility="collapsed")

    st.title("坐标工具")
    
    with st.container():
        # 模式 1: 手动
        if app_mode == "🖐️ 手动输入":
            st.subheader("手动录入")
            st.caption("配置坐标格式并输入数据。")
            
            c1, c2 = st.columns(2)
            with c1: coord_mode = st.selectbox("坐标格式", ["Decimal", "DMS", "DDM", "CGCS2000"])
            with c2:
                cm = 0
                if coord_mode == "CGCS2000":
                    cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                    cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
            
            st.divider()
            st.markdown("#### 数据编辑")
            if 'manual_df' not in st.session_state:
                st.session_state.manual_df = pd.DataFrame([{"编号": "T1", "纬度/X": "", "经度/Y": ""}, {"编号": "T2", "纬度/X": "", "经度/Y": ""}])
            edited_df = st.data_editor(st.session_state.manual_df, num_rows="dynamic", use_container_width=True)
            
            st.write("")
            if st.button("生成 KMZ 文件", type="primary"):
                log_event("Generate KMZ", "Manual")
                kml, count = generate_kmz(edited_df, coord_mode, cm)
                if count > 0:
                    kml.save("manual.kmz")
                    with open("manual.kmz", "rb") as f: st.download_button("📥 下载 KMZ", f, "manual.kmz", type="primary")
                else: st.error("请检查输入数据。")

        # 模式 2: Excel
        elif app_mode == "📊 Excel导入":
            st.subheader("Excel 导入")
            st.caption("上传文件并映射列名。")
            excel_file = st.file_uploader("上传文件", type=['xlsx', 'xls'], label_visibility="collapsed")
            if excel_file:
                try:
                    df = pd.read_excel(excel_file)
                    st.toast("✅ 文件已加载")
                    
                    st.divider()
                    st.markdown("#### 列映射")
                    cols = list(df.columns)
                    c1, c2, c3 = st.columns(3)
                    with c1: col_name = st.selectbox("编号列", ["无"] + cols)
                    with c2: col_lat = st.selectbox("纬度/X 列", cols, index=0)
                    with c3: col_lon = st.selectbox("经度/Y 列", cols, index=0)
                    
                    processed = []
                    for i, row in df.iterrows():
                        processed.append({"编号": row[col_name] if col_name != "无" else f"P{i+1}", "纬度/X": row[col_lat], "经度/Y": row[col_lon]})
                    proc_df = pd.DataFrame(processed)
                    
                    st.divider()
                    st.markdown("#### 格式确认")
                    c_set1, c_set2 = st.columns(2)
                    with c_set1: coord_mode = st.selectbox("坐标格式", ["Decimal", "DMS", "DDM", "CGCS2000"])
                    with c_set2:
                        cm = 0
                        if coord_mode == "CGCS2000":
                            cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                            cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
                    final_df = st.data_editor(proc_df, num_rows="dynamic", use_container_width=True)
                    
                    st.write("")
                    if st.button("生成 KMZ 文件", type="primary"):
                        log_event("Generate KMZ", "Excel")
                        kml, count = generate_kmz(final_df, coord_mode, cm)
                        if count > 0:
                            kml.save("excel.kmz")
                            with open("excel.kmz", "rb") as f: st.download_button("📥 下载 KMZ", f, "excel.kmz", type="primary")
                except: st.error("文件读取失败。")

        # 模式 3: AI
        elif app_mode == "📸 AI图片识别":
            st.subheader("AI 识别")
            st.caption("选取图片，AI 将自动提取坐标表格。")
            
            if 'raw_img' not in st.session_state: st.session_state.raw_img = None
            if 'ai_json_text' not in st.session_state: st.session_state.ai_json_text = ""
            if 'parsed_df' not in st.session_state: st.session_state.parsed_df = None
            
            with st.container():
                img_file = st.file_uploader("选择图片", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
            
            if img_file:
                opened_img = Image.open(img_file)
                st.session_state.raw_img = ImageOps.exif_transpose(opened_img)
                st.markdown(f'<img src="data:image/jpeg;base64,{image_to_base64(st.session_state.raw_img)}" style="width:100%; border-radius: 16px; margin-top: 16px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">', unsafe_allow_html=True)
                
                if st.button("开始识别", type="primary"):
                    log_event("AI Recognize", "Start")
                    with st.spinner("正在处理..."):
                        result = recognize_image_with_zhipu(st.session_state.raw_img)
                    if result.startswith("CRITICAL_ERROR"):
                        st.error(f"调用失败: {result}")
                    elif result.startswith("Error"):
                        st.warning(f"识别异常: {result}")
                    else:
                        clean_result = result.replace("```json", "").replace("```", "").strip()
                        st.session_state.ai_json_text = clean_result
                        try:
                            data = json.loads(clean_result)
                            st.session_state.parsed_df = pd.DataFrame(data)
                            st.toast("✅ 识别完成")
                        except: st.error("数据格式错误，请重试。")

            if st.session_state.parsed_df is not None:
                st.divider()
                st.subheader("结果核对")
                st.caption("确认坐标格式与识别结果一致。")
                
                with st.container():
                    c1, c2 = st.columns(2)
                    with c1: coord_mode = st.selectbox("图片坐标格式", ["Decimal (小数)", "DMS (度分秒)", "DDM (度.分)", "CGCS2000 (投影)"], index=0)
                    with c2:
                        cm = 0
                        if coord_mode == "CGCS2000 (投影)":
                            cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                            cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
                        else: st.empty()
                    
                    final_df = st.data_editor(st.session_state.parsed_df, num_rows="dynamic", use_container_width=True)
                    
                    st.write("")
                    if st.button("生成 KMZ 文件", type="primary"):
                        log_event("Generate KMZ", "AI Result")
                        mode_map = {"Decimal (小数)": "Decimal", "DMS (度分秒)": "DMS", "DDM (度.分)": "DDM", "CGCS2000 (投影)": "CGCS2000"}
                        kml, count = generate_kmz(final_df, mode_map[coord_mode], cm)
                        if count > 0:
                            kml.save("zhipu_result.kmz")
                            with open("zhipu_result.kmz", "rb") as f: st.download_button("📥 下载 KMZ", f, "zhipu_result.kmz", type="primary")
                        else: st.error("未生成有效数据。")
