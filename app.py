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
# 🔥 已更新为你的新 Key (2026-01-22)
ZHIPU_API_KEY = "c1bcd3c427814b0b80e8edd72205a830.mWewm9ZI2UOgwYQy"

st.set_page_config(page_title="力力的坐标工具 v21.6", page_icon="🤖", layout="centered")

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
    """
