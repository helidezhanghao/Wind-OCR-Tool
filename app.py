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
import zipfile
import xml.etree.ElementTree as ET
import tempfile
import shutil
import geopandas as gpd
import simplekml as _simplekml_mod
import sqlite3
import uuid
from pathlib import Path
from datetime import timedelta

# --- 全局配置 ---
ZHIPU_API_KEY = "c1bcd3c427814b0b80e8edd72205a830.mWewm9ZI2UOgwYQy"
# USER_PASSWORD = "2026"  <-- 普通用户密码已取消
ADMIN_PASSWORD = "0521" # 管理员密码
LOG_FILE = "usage_log.csv"
LOGO_FILENAME = "logo.png"
APP_DIR = Path(__file__).resolve().parent
FEEDBACK_BASE_DIR = Path(os.getenv("FEEDBACK_STORAGE_DIR", str(APP_DIR / "feedback_storage"))).resolve()
FEEDBACK_DB = FEEDBACK_BASE_DIR / "feedback.db"
FEEDBACK_UPLOAD_DIR = FEEDBACK_BASE_DIR / "feedback_uploads"

# 设置 layout="wide"
st.set_page_config(page_title="力力的坐标工具 v32.2", page_icon="📲", layout="wide")

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

def init_feedback_storage():
    FEEDBACK_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(FEEDBACK_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                submitted_by TEXT NOT NULL,
                user_role TEXT NOT NULL,
                app_mode TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT,
                attachment_name TEXT,
                attachment_path TEXT
            )
        """)
        cols = [row[1] for row in conn.execute("PRAGMA table_info(feedback)").fetchall()]
        if "submitted_by" not in cols:
            conn.execute("ALTER TABLE feedback ADD COLUMN submitted_by TEXT NOT NULL DEFAULT ''")
        conn.commit()

def save_feedback(submitted_by, user_role, app_mode, rating, comment="", uploaded_file=None):
    init_feedback_storage()
    attachment_name = ""
    attachment_path = ""

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".bin"
        safe_filename = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
        target_path = FEEDBACK_UPLOAD_DIR / safe_filename
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        attachment_name = uploaded_file.name
        attachment_path = str(target_path)

    with sqlite3.connect(FEEDBACK_DB) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO feedback (created_at, submitted_by, user_role, app_mode, rating, comment, attachment_name, attachment_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                str(submitted_by).strip(),
                str(user_role),
                str(app_mode),
                int(rating),
                str(comment or ""),
                attachment_name,
                attachment_path,
            )
        )
        conn.commit()
        return cur.lastrowid

def get_feedback_df():
    init_feedback_storage()
    with sqlite3.connect(FEEDBACK_DB) as conn:
        return pd.read_sql_query(
            """
            SELECT id, created_at, submitted_by, user_role, app_mode, rating, comment, attachment_name, attachment_path
            FROM feedback
            ORDER BY id DESC
            """,
            conn
        )

def delete_feedback(feedback_ids):
    init_feedback_storage()
    if not feedback_ids:
        return 0

    feedback_ids = [int(fid) for fid in feedback_ids]
    with sqlite3.connect(FEEDBACK_DB) as conn:
        placeholders = ",".join("?" for _ in feedback_ids)
        rows = conn.execute(
            f"SELECT id, attachment_path FROM feedback WHERE id IN ({placeholders})",
            feedback_ids
        ).fetchall()
        conn.execute(
            f"DELETE FROM feedback WHERE id IN ({placeholders})",
            feedback_ids
        )
        conn.commit()

    deleted = 0
    for _, attachment_path in rows:
        if attachment_path and os.path.exists(attachment_path):
            try:
                os.remove(attachment_path)
            except OSError:
                pass
        deleted += 1
    return deleted

def build_feedback_zip():
    df = get_feedback_df()
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        export_df = df.drop(columns=["attachment_path"], errors="ignore")
        zf.writestr("feedback.csv", export_df.to_csv(index=False).encode("utf-8-sig"))
        for _, row in df.iterrows():
            attachment_path = str(row.get("attachment_path", "") or "")
            attachment_name = str(row.get("attachment_name", "") or "")
            if attachment_path and os.path.exists(attachment_path):
                safe_name = attachment_name or os.path.basename(attachment_path)
                zf.write(attachment_path, arcname=f"attachments/{int(row['id'])}_{safe_name}")
    zip_buf.seek(0)
    return zip_buf

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
    # 智能列名匹配
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

def image_to_base64(image):
    buffered = BytesIO()
    if image.mode != "RGB": image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

def wgs84_to_cgcs2000(lat, lon, cm):
    """WGS84转CGCS2000投影坐标"""
    final_cm = cm if cm != 0 else int(lon / 3) * 3 + 3
    false_easting = 500000
    crs_str = f"+proj=tmerc +lat_0=0 +lon_0={final_cm} +k=1 +x_0={false_easting} +y_0=0 +ellps=GRS80 +units=m +no_defs"
    try:
        t = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_string(crs_str), always_xy=True)
        east, north = t.transform(lon, lat)
        zone_no = int(final_cm / 3)
        east_with_zone = zone_no * 1000000 + east
        return north, east, east_with_zone, zone_no, final_cm
    except:
        return None, None, None, None, None

def decimal_to_dms(deg):
    """小数度转度分秒字符串"""
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600
    return f"{d}°{m}'{s:.4f}\""

def decimal_to_ddm(deg):
    """小数度转度.分格式"""
    d = int(deg)
    m = (deg - d) * 60
    return f"{d}°{m:.6f}'"

def get_3deg_zone_info(lon, zone_override=0, cm_override=0):
    """按3度带获取带号和中央经线；可手动覆盖"""
    if zone_override:
        zone_no = int(zone_override)
        return zone_no, zone_no * 3
    if cm_override:
        final_cm = int(cm_override)
        return int(final_cm / 3), final_cm

    zone_no = int(lon / 3) + 1
    return zone_no, zone_no * 3

def parse_kmz(file_bytes):
    """解析KMZ/KML文件，返回点位列表 [{name, lat, lon}]"""
    points = []
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    def parse_kml_text(kml_text):
        try:
            root = ET.fromstring(kml_text)
        except ET.ParseError:
            return
        for pm in root.iter('{http://www.opengis.net/kml/2.2}Placemark'):
            name_el = pm.find('{http://www.opengis.net/kml/2.2}name')
            name = name_el.text.strip() if name_el is not None and name_el.text else ''
            point_el = pm.find('.//{http://www.opengis.net/kml/2.2}Point')
            if point_el is not None:
                coords_el = point_el.find('{http://www.opengis.net/kml/2.2}coordinates')
                if coords_el is not None and coords_el.text:
                    parts = coords_el.text.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            lon, lat = float(parts[0]), float(parts[1])
                            points.append({'name': name, 'lat': lat, 'lon': lon})
                        except:
                            pass

    # 判断是KMZ还是KML
    try:
        with zipfile.ZipFile(BytesIO(file_bytes)) as z:
            for fname in z.namelist():
                if fname.endswith('.kml'):
                    parse_kml_text(z.read(fname).decode('utf-8', errors='ignore'))
    except zipfile.BadZipFile:
        # 直接当KML处理
        parse_kml_text(file_bytes.decode('utf-8', errors='ignore'))

    return points

def build_excel(points, output_format, cm=0, zone_override=0):
    """根据选择的格式生成Excel字节"""
    rows = []
    for p in points:
        lat, lon = p['lat'], p['lon']
        name = p['name']
        if output_format == "WGS84 小数度":
            rows.append({"编号": name, "纬度(°)": round(lat, 8), "经度(°)": round(lon, 8)})
        elif output_format == "WGS84 度分秒(DMS)":
            rows.append({"编号": name, "纬度": decimal_to_dms(lat), "经度": decimal_to_dms(lon)})
        elif output_format == "WGS84 度.分(DDM)":
            rows.append({"编号": name, "纬度": decimal_to_ddm(lat), "经度": decimal_to_ddm(lon)})
        elif output_format == "CGCS2000 投影坐标":
            use_zone_no, use_cm = get_3deg_zone_info(lon, zone_override, cm)
            north, east, east_with_zone, zone_no, final_cm = wgs84_to_cgcs2000(lat, lon, use_cm)
            if north is not None:
                rows.append({
                    "编号": name,
                    "X(北)": round(north, 3),
                    "Y(东,带号)": round(east_with_zone, 3),
                    "Y(东,不带号)": round(east, 3),
                    "带号": use_zone_no,
                    "中央经线": use_cm
                })
        elif output_format == "全格式(所有列)":
            use_zone_no, use_cm = get_3deg_zone_info(lon, zone_override, cm)
            north, east, east_with_zone, zone_no, final_cm = wgs84_to_cgcs2000(lat, lon, use_cm)
            rows.append({
                "编号": name,
                "纬度_小数": round(lat, 8),
                "经度_小数": round(lon, 8),
                "纬度_DMS": decimal_to_dms(lat),
                "经度_DMS": decimal_to_dms(lon),
                "纬度_DDM": decimal_to_ddm(lat),
                "经度_DDM": decimal_to_ddm(lon),
                "X_CGCS2000(北)": round(north, 3) if north is not None else "",
                "Y_CGCS2000(东,带号)": round(east_with_zone, 3) if east_with_zone is not None else "",
                "Y_CGCS2000(东,不带号)": round(east, 3) if east is not None else "",
                "带号": use_zone_no,
                "中央经线": use_cm,
            })

    df = pd.DataFrame(rows)
    buf = BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf, df

LAND_TYPE_COLORS = [
    simplekml.Color.changealpha('aa', simplekml.Color.red),
    simplekml.Color.changealpha('aa', simplekml.Color.blue),
    simplekml.Color.changealpha('aa', simplekml.Color.green),
    simplekml.Color.changealpha('aa', simplekml.Color.yellow),
    simplekml.Color.changealpha('aa', simplekml.Color.orange),
    simplekml.Color.changealpha('aa', simplekml.Color.purple),
    simplekml.Color.changealpha('aa', simplekml.Color.cyan),
    simplekml.Color.changealpha('aa', simplekml.Color.magenta),
    simplekml.Color.changealpha('aa', simplekml.Color.lime),
    simplekml.Color.changealpha('aa', simplekml.Color.pink),
    simplekml.Color.changealpha('aa', simplekml.Color.teal),
    simplekml.Color.changealpha('aa', simplekml.Color.brown),
    simplekml.Color.changealpha('aa', simplekml.Color.coral),
    simplekml.Color.changealpha('aa', simplekml.Color.gold),
]

def find_land_type_col(gdf):
    """自动识别地类名称字段"""
    keywords = ['地类名称', '地类', 'DLMC', 'dlmc', 'LAND_TYPE', 'land_type', 'TYPENAME', 'NAME', 'name']
    for col in gdf.columns:
        if col in keywords:
            return col
    for col in gdf.columns:
        for kw in ['地类', 'DLMC', 'dlmc', 'land', 'type']:
            if kw.lower() in col.lower():
                return col
    return None

def geom_to_kml_folder(folder, geom, name, style):
    """将几何体写入KML folder"""
    if geom is None or geom.is_empty:
        return
    gt = geom.geom_type
    if gt == 'Polygon':
        pol = folder.newpolygon(name=name)
        pol.outerboundaryis = list(geom.exterior.coords)
        pol.style = style
    elif gt == 'MultiPolygon':
        for i, part in enumerate(geom.geoms):
            pol = folder.newpolygon(name=f"{name}_{i}")
            pol.outerboundaryis = list(part.exterior.coords)
            pol.style = style
    elif gt == 'Point':
        pnt = folder.newpoint(name=name)
        pnt.coords = [(geom.x, geom.y)]
        pnt.style = style
    elif gt == 'MultiPoint':
        for i, part in enumerate(geom.geoms):
            pnt = folder.newpoint(name=f"{name}_{i}")
            pnt.coords = [(part.x, part.y)]
            pnt.style = style
    elif gt == 'LineString':
        ls = folder.newlinestring(name=name)
        ls.coords = list(geom.coords)
        ls.style = style
    elif gt == 'MultiLineString':
        for i, part in enumerate(geom.geoms):
            ls = folder.newlinestring(name=f"{name}_{i}")
            ls.coords = list(part.coords)
            ls.style = style

def classify_to_kmz_zip(uploaded_files, land_col_override=None):
    """
    读取上传的文件（支持KMZ/KML/Shapefile/GeoJSON等），
    按地类名称分类，返回 (zip_bytes, 分类统计列表, 错误信息)
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        # 保存上传文件
        file_names = []
        for f in uploaded_files:
            path = os.path.join(tmp_dir, f.name)
            with open(path, 'wb') as out:
                out.write(f.read())
            file_names.append(f.name)

        # 读取数据
        gdf = None
        shp_files = [n for n in file_names if n.lower().endswith('.shp')]
        kmz_files = [n for n in file_names if n.lower().endswith(('.kmz', '.kml'))]
        geojson_files = [n for n in file_names if n.lower().endswith(('.geojson', '.json'))]

        if shp_files:
            gdf = gpd.read_file(os.path.join(tmp_dir, shp_files[0]))
        elif kmz_files:
            path = os.path.join(tmp_dir, kmz_files[0])
            try:
                gdf = gpd.read_file(path, driver='KML')
            except:
                gdf = gpd.read_file(f"/{path}", driver='LIBKML')
        elif geojson_files:
            gdf = gpd.read_file(os.path.join(tmp_dir, geojson_files[0]))
        else:
            # 尝试读取任意文件
            for fname in file_names:
                try:
                    gdf = gpd.read_file(os.path.join(tmp_dir, fname))
                    break
                except:
                    continue

        if gdf is None or len(gdf) == 0:
            return None, [], "无法读取数据，请检查文件格式"

        # 转WGS84
        if gdf.crs is None:
            return None, [], "文件缺少坐标系信息（.prj），无法处理"
        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        # 找地类字段
        land_col = land_col_override if land_col_override else find_land_type_col(gdf)
        if land_col is None:
            return None, [], f"未找到地类名称字段，现有字段：{', '.join(gdf.columns.tolist())}"

        categories = gdf[land_col].dropna().unique().tolist()
        if not categories:
            return None, [], "地类字段为空"

        # 生成KMZ
        kmz_dir = os.path.join(tmp_dir, 'kmz_out')
        os.makedirs(kmz_dir, exist_ok=True)
        result_stats = []

        for i, cat in enumerate(categories):
            subset = gdf[gdf[land_col] == cat]
            if subset.empty:
                continue

            kml = simplekml.Kml(name=str(cat))
            folder = kml.newfolder(name=str(cat))
            color = LAND_TYPE_COLORS[i % len(LAND_TYPE_COLORS)]
            style = simplekml.Style()
            style.polystyle.color = color
            style.polystyle.outline = 1
            style.linestyle.color = simplekml.Color.white
            style.linestyle.width = 0.5
            style.iconstyle.color = color

            count = 0
            for _, row in subset.iterrows():
                geom = row.geometry
                geom_to_kml_folder(folder, geom, str(cat), style)
                count += 1

            safe = str(cat).replace('/', '_').replace('\\', '_').replace(' ', '_')
            kml.savekmz(os.path.join(kmz_dir, f"{safe}.kmz"))
            result_stats.append(f"{cat}（{count}个图斑）")

        # 打包zip
        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(kmz_dir):
                zf.write(os.path.join(kmz_dir, fname), fname)
        zip_buf.seek(0)
        return zip_buf, result_stats, None

    except Exception as e:
        return None, [], str(e)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

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

# --- 1. 登录界面 (居中布局 + 免密直连) ---
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
                # 🔥🔥🔥 修改点：点击直接进入，不再跳转输入密码 🔥🔥🔥
                if st.button("🚀 普通用户登录", type="primary", use_container_width=True):
                    st.session_state.user_role = 'user'
                    log_event("Login", "User Auto-Login")
                    st.rerun()
                
                st.write("")
                
                # 管理员依然需要密码
                if st.button("🛡️ 管理员登录", use_container_width=True):
                    st.session_state.login_mode = 'admin_input'
                    st.rerun()

        # 普通用户的 user_input 状态代码已移除

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

    st.divider()
    st.subheader("📝 用户反馈")
    feedback_df = get_feedback_df()
    cfb1, cfb2, cfb3 = st.columns(3)
    with cfb1:
        st.metric("反馈总数", len(feedback_df))
    with cfb2:
        avg_rating = round(feedback_df["rating"].mean(), 2) if not feedback_df.empty else 0
        st.metric("平均评分", avg_rating)
    with cfb3:
        with_attachments = int(feedback_df["attachment_name"].fillna("").astype(str).ne("").sum()) if not feedback_df.empty else 0
        st.metric("带附件反馈", with_attachments)

    feedback_zip = build_feedback_zip()
    st.download_button(
        "📦 一键打包下载全部反馈",
        feedback_zip,
        file_name=f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
        type="primary"
    )

    if feedback_df.empty:
        st.info("暂时还没有用户反馈。")
    else:
        st.markdown("#### 筛选")
        sf1, sf2, sf3 = st.columns(3)
        with sf1:
            rating_filter = st.multiselect("按评分筛选", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
        with sf2:
            date_mode = st.selectbox("按时间筛选", ["全部", "近7天", "近30天", "自定义"])
        with sf3:
            name_keyword = st.text_input("按姓名/留言搜索")

        filtered_df = feedback_df.copy()
        if rating_filter:
            filtered_df = filtered_df[filtered_df["rating"].isin(rating_filter)]
        if date_mode != "全部":
            filtered_df["created_at_dt"] = pd.to_datetime(filtered_df["created_at"], errors="coerce")
            now = datetime.now()
            if date_mode == "近7天":
                filtered_df = filtered_df[filtered_df["created_at_dt"] >= now - timedelta(days=7)]
            elif date_mode == "近30天":
                filtered_df = filtered_df[filtered_df["created_at_dt"] >= now - timedelta(days=30)]
            elif date_mode == "自定义":
                d1, d2 = st.columns(2)
                with d1:
                    start_date = st.date_input("开始日期", value=(now - timedelta(days=30)).date(), key="fb_start_date")
                with d2:
                    end_date = st.date_input("结束日期", value=now.date(), key="fb_end_date")
                filtered_df = filtered_df[
                    (filtered_df["created_at_dt"] >= pd.Timestamp(start_date)) &
                    (filtered_df["created_at_dt"] < pd.Timestamp(end_date) + pd.Timedelta(days=1))
                ]
            filtered_df = filtered_df.drop(columns=["created_at_dt"], errors="ignore")
        if name_keyword.strip():
            keyword = name_keyword.strip()
            filtered_df = filtered_df[
                filtered_df["submitted_by"].astype(str).str.contains(keyword, case=False, na=False) |
                filtered_df["comment"].astype(str).str.contains(keyword, case=False, na=False)
            ]

        show_df = feedback_df.copy()
        show_df = filtered_df.copy()
        show_df["删除"] = False
        show_df = show_df.rename(columns={
            "id": "ID",
            "created_at": "提交时间",
            "submitted_by": "姓名",
            "user_role": "角色",
            "app_mode": "功能模块",
            "rating": "评分",
            "comment": "意见",
            "attachment_name": "附件名",
            "attachment_path": "附件路径",
        })
        edited_feedback_df = st.data_editor(
            show_df[["删除", "ID", "提交时间", "姓名", "角色", "功能模块", "评分", "意见", "附件名", "附件路径"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "删除": st.column_config.CheckboxColumn("删除"),
                "ID": st.column_config.NumberColumn("ID", disabled=True),
                "提交时间": st.column_config.TextColumn("提交时间", disabled=True),
                "姓名": st.column_config.TextColumn("姓名", disabled=True),
                "角色": st.column_config.TextColumn("角色", disabled=True),
                "功能模块": st.column_config.TextColumn("功能模块", disabled=True),
                "评分": st.column_config.NumberColumn("评分", disabled=True),
                "意见": st.column_config.TextColumn("意见", disabled=True),
                "附件名": st.column_config.TextColumn("附件名", disabled=True),
                "附件路径": st.column_config.TextColumn("附件路径", disabled=True),
            }
        )
        if st.button("🗑️ 删除勾选反馈", use_container_width=True):
            selected_ids = edited_feedback_df.loc[edited_feedback_df["删除"] == True, "ID"].tolist()
            deleted = delete_feedback(selected_ids)
            if deleted:
                st.success(f"已删除 {deleted} 条反馈。")
                st.rerun()
            else:
                st.warning("请先勾选要删除的反馈。")

        st.markdown("#### 附件预览")
        preview_candidates = filtered_df[filtered_df["attachment_path"].fillna("").astype(str) != ""]
        if preview_candidates.empty:
            st.caption("当前筛选结果中没有可预览附件。")
        else:
            preview_map = {
                f"{int(row['id'])} | {row['submitted_by']} | {row['attachment_name']}": row
                for _, row in preview_candidates.iterrows()
            }
            selected_preview_key = st.selectbox("选择一条带附件的反馈", list(preview_map.keys()))
            preview_row = preview_map[selected_preview_key]
            attachment_path = str(preview_row["attachment_path"])
            attachment_name = str(preview_row["attachment_name"])
            if os.path.exists(attachment_path):
                suffix = Path(attachment_name).suffix.lower()
                st.write(f"反馈人：{preview_row['submitted_by']}  |  评分：{preview_row['rating']}")
                st.write(f"留言：{preview_row['comment']}")
                if suffix in [".png", ".jpg", ".jpeg"]:
                    st.image(attachment_path, caption=attachment_name, use_container_width=True)
                elif suffix == ".pdf":
                    with open(attachment_path, "rb") as f:
                        pdf_bytes = f.read()
                    st.download_button(
                        "📥 下载该 PDF 附件",
                        pdf_bytes,
                        file_name=attachment_name,
                        mime="application/pdf"
                    )
                else:
                    with open(attachment_path, "rb") as f:
                        raw_bytes = f.read()
                    st.download_button(
                        "📥 下载该附件",
                        raw_bytes,
                        file_name=attachment_name,
                        mime="application/octet-stream"
                    )
            else:
                st.warning("附件文件不存在，可能已被删除或迁移。")


# --- 3. 普通用户界面 ---
elif st.session_state.user_role == 'user':
    
    with st.sidebar:
        if st.button("🔒 退出登录"):
            st.session_state.user_role = None
            st.rerun() 
        st.divider()
        app_mode = st.radio("功能选择", ["🖐️ 手动输入", "📄 文本导入", "📸 AI图片识别", "📂 KMZ转Excel"], index=2)
        st.info("切换模式会清空当前数据")
        st.divider()
        st.markdown("### 📝 反馈意见")
        feedback_name = st.text_input("你的姓名", key="feedback_name")
        feedback_rating = st.slider("工具评分", min_value=1, max_value=5, value=5, step=1, key="feedback_rating")
        feedback_comment = st.text_area("留言 / 修改建议", height=120, key="feedback_comment")
        feedback_file = st.file_uploader(
            "上传附图（可选）",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            key="feedback_file"
        )
        if st.button("📨 提交反馈", use_container_width=True):
            if not str(feedback_name).strip():
                st.error("请先填写姓名。")
            elif not str(feedback_comment).strip() and feedback_file is None:
                st.error("请至少填写留言或上传一张附图。")
            else:
                feedback_id = save_feedback(
                    submitted_by=feedback_name,
                    user_role="user",
                    app_mode=app_mode,
                    rating=feedback_rating,
                    comment=feedback_comment,
                    uploaded_file=feedback_file
                )
                log_event("Submit Feedback", f"ID={feedback_id}")
                st.success("反馈已提交，我这边可以在管理员后台查看。")

    st.title("力力的坐标工具 v32.2")
    
    # 模式 1: 手动
    if app_mode == "🖐️ 手动输入":
        st.header("🖐️ 手动录入")
        c1, c2 = st.columns(2)
        with c1: coord_mode_display = st.selectbox("坐标格式", ["Decimal (小数)", "DMS (度分秒)", "DDM (度.分)", "CGCS2000 (投影)"])
        with c2:
            cm = 0
            if "CGCS2000" in coord_mode_display:
                cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
        
        if 'manual_df' not in st.session_state:
            st.session_state.manual_df = pd.DataFrame([{"编号": "T1", "纬度/X": "", "经度/Y": ""}, {"编号": "T2", "纬度/X": "", "经度/Y": ""}])
        edited_df = st.data_editor(st.session_state.manual_df, num_rows="dynamic", use_container_width=True)
        
        if st.button("🚀 生成 KMZ", type="primary"):
            log_event("Generate KMZ", "Manual")
            mode_map = {"Decimal (小数)": "Decimal", "DMS (度分秒)": "DMS", "DDM (度.分)": "DDM", "CGCS2000 (投影)": "CGCS2000"}
            kml, count = generate_kmz(edited_df, mode_map[coord_mode_display], cm)
            if count > 0:
                kml.save("manual.kmz")
                with open("manual.kmz", "rb") as f: st.download_button("📥 下载文件", f, "manual.kmz", type="primary")
            else: st.error("数据无效")

    # 模式 2: 文本导入
    elif app_mode == "📄 文本导入":
        st.header("📄 文本导入 (Excel/TXT/CSV)")
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
                with c_set1: coord_mode_display = st.selectbox("坐标格式", ["Decimal (小数)", "DMS (度分秒)", "DDM (度.分)", "CGCS2000 (投影)"])
                with c_set2:
                    cm = 0
                    if "CGCS2000" in coord_mode_display:
                        cm_ops = {0:0, 75:75, 81:81, 87:87, 93:93, 99:99, 105:105, 114:114, 123:123}
                        cm = st.selectbox("中央经线", list(cm_ops.keys()), format_func=lambda x: "自动" if x==0 else str(x))
                final_df = st.data_editor(proc_df, num_rows="dynamic", use_container_width=True)
                
                if st.button("🚀 生成 KMZ", type="primary"):
                    log_event("Generate KMZ", "Text Import")
                    mode_map = {"Decimal (小数)": "Decimal", "DMS (度分秒)": "DMS", "DDM (度.分)": "DDM", "CGCS2000 (投影)": "CGCS2000"}
                    kml, count = generate_kmz(final_df, mode_map[coord_mode_display], cm)
                    if count > 0:
                        kml.save("text_import.kmz")
                        with open("text_import.kmz", "rb") as f: st.download_button("📥 下载文件", f, "text_import.kmz", type="primary")
            except Exception as e: st.error(f"读取失败: {str(e)}")

    # 模式 3: AI
    elif app_mode == "📸 AI图片识别":
        st.header("📸 AI 识别")
        if 'raw_img' not in st.session_state: st.session_state.raw_img = None
        if 'ai_json_text' not in st.session_state: st.session_state.ai_json_text = ""
        if 'parsed_df' not in st.session_state: st.session_state.parsed_df = None
        
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

    # 模式 4: KMZ转Excel
    elif app_mode == "📂 KMZ转Excel":
        st.header("📂 KMZ / KML 转 Excel")
        kmz_file = st.file_uploader("上传 KMZ 或 KML 文件", type=['kmz', 'kml'])
        if kmz_file:
            points = parse_kmz(kmz_file.read())
            if not points:
                st.error("未读取到点位数据，请确认文件包含点标注")
            else:
                st.success(f"共读取到 {len(points)} 个点位")
                st.dataframe(pd.DataFrame(points).rename(columns={'name':'编号','lat':'纬度','lon':'经度'}), use_container_width=True)
                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    output_format = st.selectbox("输出坐标格式", [
                        "全格式(所有列)",
                        "WGS84 小数度",
                        "WGS84 度分秒(DMS)",
                        "WGS84 度.分(DDM)",
                        "CGCS2000 投影坐标",
                    ])
                with c2:
                    cm = 0
                    zone_override = 0
                    if output_format in ["CGCS2000 投影坐标", "全格式(所有列)"]:
                        zone_mode = st.selectbox("度带号模式", ["自动判定", "手动指定"])
                        if zone_mode == "手动指定":
                            zone_override = st.selectbox("3度带号", list(range(25, 46)))
                            cm = zone_override * 3
                            st.caption(f"当前中央经线：{cm}")
                        else:
                            st.caption("将根据点位经度自动判定3度带号和中央经线")

                if st.button("🚀 生成 Excel", type="primary"):
                    log_event("KMZ to Excel", output_format)
                    buf, df = build_excel(points, output_format, cm, zone_override)
                    st.dataframe(df, use_container_width=True)
                    st.download_button("📥 下载 Excel", buf, file_name="坐标导出.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       type="primary")
