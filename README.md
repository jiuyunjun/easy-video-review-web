# 本地图片/视频分享站 – README

特性：
- 响应式 UI，兼容电脑与手机
- 支持拖拽和多线程上传
- 相册页面可按名称、时间或大小排序并支持升/降序切换，缩略图下方依次显示文件大小、上传时间和创建时间
- 模板文件位于 `templates/` 目录，可自行修改

## 依赖安装
### 必选
```bash
pip install Flask Pillow
```

### 可选
| 功能 | 包 | 安装命令 |
|------|----|-----------|
| RAW 预览 | rawpy imageio | `pip install rawpy imageio` |
| 视频缩略图 | opencv-python | `pip install opencv-python` |
| 自动分场景 | scenedetect opencv-python | `pip install scenedetect opencv-python` |
| 生产部署 | gunicorn | `pip install gunicorn` |

## 快速运行
```bash
python EasyReview.py
# 浏览器访问 http://<本机或局域网IP>:5123
```

### 运行测试
```bash
python -m pytest -v
```

## 目录说明
- **uploads/** 上传文件根目录，子文件夹即相册名  
- **.thumbs/** 每个相册下自动生成的缩略图缓存  

> 默认单文件大小上限 5 GB，可在 `EasyReview.py` 顶部 `MAX_CONTENT_LENGTH` 修改。

## License
MIT
