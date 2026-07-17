# Wind OCR Tool

一个轻量的 Streamlit 坐标工具，支持坐标录入、图片识别、KMZ 生成、KMZ/KML 转 Excel，以及 WGS84/UTM 双向转换。

## 主要功能

- 手动或通过 Excel、CSV、TXT 导入坐标并生成 KMZ
- 图片表格识别后生成 KMZ
- KMZ/KML 点位导出为多种坐标格式的 Excel
- WGS84 与 UTM 双向批量转换（UTM 1-60 带，南/北半球）
- WGS84、DMS、DDM、CGCS2000 和 UTM 坐标支持
- 简单的使用日志、用户反馈和管理员后台

## 本地运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 配置

复制示例文件并填写自己的新凭据：

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

也可以通过环境变量设置：

- `ZHIPU_API_KEY`：图片识别使用的智谱 API Key
- `ADMIN_PASSWORD`：管理员后台密码

不要把 `.streamlit/secrets.toml`、API Key 或真实密码提交到 GitHub。已经公开过的凭据必须先在对应平台撤销并重新生成。

## UTM 使用说明

- WGS84 转 UTM 时可以自动判断带号和半球，也可以手动指定。
- UTM 转 WGS84 时必须选择原坐标对应的 UTM 带号与半球。
- 东坐标和北坐标的单位为米。

## 测试

```bash
python3 -m unittest discover -s tests -v
```
