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

st.set_page_config(page_title="风资源坐标神器v2.0", page_icon="⚡", layout="centered")

# --- 核心逻辑 ---
def preprocess_image(image):
    """图像增强：灰度 -> 提高对比度 -> 二值化，专门拯救渣画质"""
    img = ImageOps.grayscale(image)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0) # 提高对比度
    # 二值化处理（让字变黑，背景变全白）
    thresh = 200
    fn = lambda x : 255 if x > thresh else 0
    return img.point(fn, mode='1')

def extract_numbers(text):
    """暴力提取：不管中间夹杂什么乱码，只抓取数字和点"""
    # 这一行正则意思是：匹配所有整数或小数
    # 比如 "X: 123.456 | Y: 88.9" -> ['123.456', '88.9']
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    # 过滤掉奇怪的单个数字（比如行号1, 2）或者太短的数字
    valid_nums = []
    for n in nums:
        # OCR常把竖线识别为1，这里过滤掉纯整数且长度小于2的（大概率是杂音）
        if '.' not in n and len(n) < 3: 
            continue
        valid_nums.append(float(n))
    return valid_nums

def cgcs2000_to_wgs84(v1, v2, cm_val, force_swap):
    # 这里的逻辑和之前一样，负责数学转换
    x, y = (v2, v1) if force_swap else (v1, v2)
    
    # 智能判断谁是Y（带号的那个通常是Y）
    if 10000000 < x < 100000000: # 如果x像带号坐标
         x, y = y, x # 换一下

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
    except: return None, "转换计算错"

# --- 界面 ---
st.title("⚡ 风资源坐标神器 v2.0")
st.caption("增强图像处理 + 表格修正模式")

# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 坐标参数")
    cm_options = {
        "自动识别(8位带号)": 0, "新疆西 (75)": 75, "新疆中 (81)": 81, "新疆东 (87)": 87,
        "甘肃/青海 (93)": 93, "内蒙西 (99)": 99, "内蒙中 (105)": 105,
        "晋/陕 (111)": 111, "张家口 (114)": 114, "京/承 (117)": 117, "东北 (123)": 123
    }
    cm_val = cm_options[st.selectbox("大地2000区域", list(cm_options.keys()))]
    force_swap = st.checkbox("强制交换 XY (纵轴为X时勾选)", value=False)

# 1. 上传与识别
img_file = st.file_uploader("📸 拍照或传图", type=['png', 'jpg', 'jpeg'])
raw_data = []

if img_file:
    # 显示原图
    image = Image.open(img_file)
    
    # 图像处理预览
    processed_img = preprocess_image(image)
    with st.expander("👀 查看图像增强效果"):
        st.image(processed_img, caption="机器看到的图（黑白高对比）", use_column_width=True)

    if st.button('🔥 开始强力识别'):
        with st.spinner('正在逐行扫描...'):
            # OCR 识别
            text = pytesseract.image_to_string(processed_img, lang='eng', config='--psm 6')
            lines = text.split('\n')
            
            # 智能解析每一行
            for line in lines:
                nums = extract_numbers(line)
                # 只有当一行里恰好提取出2个或3个有效数字时，才认为是坐标
                if len(nums) >= 2:
                    # 取前两个最大的可能是坐标
                    raw_data.append({"值1": nums[0], "值2": nums[1], "备注": "OCR识别"})
            
            if not raw_data:
                st.error("识别失败，画面太乱或没找到数字。请手动录入👇")
            else:
                st.success(f"成功抓取 {len(raw_data)} 行数据！请在下方表格检查核对。")

# 2. 数据编辑区 (这是 v2.0 的核心)
st.subheader("📝 数据核对与生成")
st.info("👇 这里可以直接修改数字！改完直接点生成。")

# 初始化表格数据
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["值1", "值2", "备注"])

# 如果有新识别的数据，覆盖进去
if raw_data:
    st.session_state.df = pd.DataFrame(raw_data)
    # 清空一下以免重复添加
    raw_data = []

# 显示可编辑表格
edited_df = st.data_editor(st.session_state.df, num_rows="dynamic", key="editor")

# 3. 生成逻辑
if st.button("🚀 生成最终 KMZ"):
    kml = simplekml.Kml()
    valid_count = 0
    
    for index, row in edited_df.iterrows():
        try:
            v1, v2 = float(row["值1"]), float(row["值2"])
            lat, lon = 0, 0
            
            # 判断是经纬度还是大地2000
            if v1 < 180 and v2 < 180: # 经纬度
                lat, lon = (v1, v2) if v1 < v2 else (v2, v1)
            else: # 大地2000
                res, msg = cgcs2000_to_wgs84(v1, v2, cm_val, force_swap)
                if res: lat, lon = res, msg
                else: continue # 转换失败跳过
            
            # 添加点
            kml.newpoint(name=f"P{index+1}", coords=[(lon, lat)])
            valid_count += 1
        except:
            continue

    if valid_count > 0:
        st.success(f"✅ 成功生成 {valid_count} 个点！")
        kml.save("out.kmz")
        with open("out.kmz", "rb") as f:
            st.download_button("📥 点击下载 KMZ", f, "Project.kmz")
    else:
        st.warning("表格是空的，或者数据格式不对（必须是数字）。")
