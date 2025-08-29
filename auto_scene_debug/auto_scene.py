# detect_and_plot.py
from pathlib import Path
import csv
import os
import math
import matplotlib.pyplot as plt

from scenedetect import open_video, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector

# ==== 可调参数 ====
VIDEO_PATH = Path("test.mp4")         # 改成你的文件名
DETECT_THRESHOLD = 5.0                # 场景检测阈值(传给ContentDetector)
DIFF_THRESHOLD = 5.0                  # 场景内部最大content_val低于此阈值则导出中间帧
OUT_DIR = Path("scene_mids_lowdiff")  # 导出目录
# ==================

assert VIDEO_PATH.exists(), f"找不到视频: {VIDEO_PATH.resolve()}"

# 1) 构建 SceneManager，并挂 StatsManager（0.6.x 推荐）
stats_manager = StatsManager()
scene_manager = SceneManager(stats_manager=stats_manager)
scene_manager.add_detector(ContentDetector(threshold=DETECT_THRESHOLD))

# 2) 运行检测
video = open_video(str(VIDEO_PATH))
scene_manager.detect_scenes(video=video)
scenes = scene_manager.get_scene_list()

# 3) 导出逐帧指标到 CSV
csv_out = VIDEO_PATH.with_suffix(".stats.csv")
scene_manager.stats_manager.save_to_csv(csv_out.open("w", encoding="utf-8"))
print(f"已导出逐帧指标: {csv_out}")

# ---- 从这里开始：读取 CSV，画「Time vs content_val」 ----

def parse_timecode_to_seconds(tc: str) -> float:
    """支持 HH:MM:SS(.ms) / MM:SS(.ms) / SS(.ms)"""
    tc = tc.strip()
    parts = tc.split(":")
    if len(parts) == 3:
        h = int(parts[0]); m = int(parts[1]); s = float(parts[2])
    elif len(parts) == 2:
        h = 0; m = int(parts[0]); s = float(parts[1])
    else:
        h = 0; m = 0; s = float(parts[0])
    return h * 3600 + m * 60 + s

def seconds_to_timecode(sec: float) -> str:
    """转换为 00:00:00.000 格式"""
    if sec < 0: sec = 0.0
    ms = int(round((sec - math.floor(sec)) * 1000))
    s = int(sec) % 60
    m = (int(sec) // 60) % 60
    h = int(sec) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

times, cvals = [], []
with csv_out.open(newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)

    # 兼容不同版本的列名（有的写 Content Value，有的写 content_val）
    field_map = {name.strip().lower().replace(" ", "_"): name for name in r.fieldnames}
    time_col = field_map.get("timecode")          # "Timecode"
    cval_col = field_map.get("content_val") or field_map.get("content_value")  # "content_val" 或 "Content Value"

    assert time_col and cval_col, f"找不到列名，实际表头为：{r.fieldnames}"

    for row in r:
        tc = row.get(time_col, "").strip()
        cv = row.get(cval_col, "").strip()
        if not tc or not cv:
            continue
        try:
            times.append(parse_timecode_to_seconds(tc))
            cvals.append(float(cv))
        except ValueError:
            # 跳过无法解析的行
            continue

# 画曲线：x=时间(秒)，y=content_val
plt.plot(times, cvals)
plt.xlabel("Time (s)")
plt.ylabel("content_val")
plt.title("PySceneDetect: content_val vs Time")

# 把检测到的切点用竖线标出来
for (start_tc, end_tc) in scenes:
    cut_sec = start_tc.get_seconds()
    plt.axvline(cut_sec, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
# ---- 新版：导出“低差异度帧集合”的中间帧 ----
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import cv2
except ImportError as e:
    raise SystemExit("需要安装 opencv-python：pip install opencv-python") from e

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise SystemExit(f"无法打开视频：{VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
if fps <= 0:
    fps = float(video.base_timecode.framerate.num) / float(video.base_timecode.framerate.den)

import numpy as np
t_arr = np.array(times, dtype=float)   # 每帧时间(秒)
v_arr = np.array(cvals, dtype=float)   # 每帧 content_val

exported = 0
for idx, (start_tc, end_tc) in enumerate(scenes, start=1):
    start_s = start_tc.get_seconds()
    end_s = end_tc.get_seconds()
    if end_s <= start_s:
        continue

    # 该场景内 & 低差异度 的帧集合：content_val < DIFF_THRESHOLD
    mask_scene = (t_arr >= start_s) & (t_arr < end_s)
    mask_low   = mask_scene & (v_arr < DIFF_THRESHOLD)

    if not np.any(mask_low):
        # 该场景没有“低差异度”帧，跳过
        continue

    # 按时间排序的低差异度帧集合（t_arr 本就按时间递增，mask会保持顺序）
    low_times = t_arr[mask_low]
    # 取集合的中位（中间）帧
    mid_idx = len(low_times) // 2
    mid_s = float(low_times[mid_idx])

    # 把该帧导出为图片（失败则兜底到该场景第一帧）
    mid_frame = int(round(mid_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start_s * fps)))
        ok, frame = cap.read()
    if not ok or frame is None:
        print(f"[跳过] 场景{idx:04d} 读取帧失败。")
        continue

    # 文件名：包含场景序号 & 选中帧的时间码
    def seconds_to_timecode(sec: float) -> str:
        import math
        ms = int(round((sec - math.floor(sec)) * 1000))
        s = int(sec) % 60
        m = (int(sec) // 60) % 60
        h = int(sec) // 3600
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    mid_tc_str = seconds_to_timecode(mid_s)
    out_name = f"scene-{idx:04d}-lowdiff-mid-{mid_tc_str.replace(':','-').replace('.','_')}.png"
    out_path = OUT_DIR / out_name
    import os
    cv2.imwrite(str(out_path), frame)
    exported += 1
    print(f"[导出] 场景{idx:04d} | 低差异度帧数={len(low_times)} | 选中 mid={mid_tc_str} -> {out_path.name}")

cap.release()
print(f"完成：共导出 {exported} 张（按“低差异度集合中位帧”）到：{OUT_DIR}/")
print(f"筛选条件：content_val < {DIFF_THRESHOLD}")
