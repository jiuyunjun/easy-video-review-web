# detect_plot.py
'''
python detect_plot.py --video test.mp4 --detect-th 5 --diff-th 5 --out-dir scene_mids_lowdiff --fig-w 16 --fig-h 9 --dpi 300 --show
'''
# from pathlib import Path
# from detect_plot import PipelineConfig, run_pipeline

# cfg = PipelineConfig(
#     video_path=Path("test.mp4"),
#     detect_threshold=5.0,
#     diff_threshold=5.0,
#     out_dir=Path("scene_mids_lowdiff"),
#     fig_size=(16, 9),
#     dpi=300,
#     show_plot=False,
# )
# result = run_pipeline(cfg)
# print(result["png"], result["exported_count"])

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional

import os
import math
import csv

# 后端切换（无显示环境时）
import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # 允许仅做分析 / 绘图而不导帧

from scenedetect import open_video, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector


# =========================
# 工具函数
# =========================
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


# =========================
# 配置
# =========================
@dataclass
class PipelineConfig:
    video_path: Path
    detect_threshold: float = 5.0     # ContentDetector 阈值
    diff_threshold: float = 5.0       # content_val < diff_threshold 的“低差异度”集合
    out_dir: Path = Path("scene_mids_lowdiff")

    # 绘图相关
    fig_size: Tuple[float, float] = (16.0, 9.0)
    dpi: int = 300
    show_plot: bool = False           # 是否 plt.show()
    save_png: bool = True
    save_svg: bool = True

    # 其他
    verbose: bool = True


# =========================
# 核心步骤函数（可单独复用）
# =========================
def detect_scenes(video_path: Path, detect_threshold: float) -> Tuple[List[Tuple], Path, object]:
    """
    运行 PySceneDetect 场景检测，返回：
    - scenes: [(start_tc, end_tc), ...]
    - csv_out: 指标 CSV 路径
    - video: open_video 返回的视频对象（供帧率兜底）
    """
    assert video_path.exists(), f"找不到视频: {video_path.resolve()}"

    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager=stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=detect_threshold))

    video = open_video(str(video_path))
    scene_manager.detect_scenes(video=video)
    scenes = scene_manager.get_scene_list()

    csv_out = video_path.with_suffix(".stats.csv")
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        scene_manager.stats_manager.save_to_csv(f)

    return scenes, csv_out, video


def load_stats(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 PySceneDetect 导出的 CSV 读取时间与 content_val。
    返回 (t_arr, v_arr)
    """
    times: List[float] = []
    cvals: List[float] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise RuntimeError(f"CSV 没有表头: {csv_path}")

        field_map = {name.strip().lower().replace(" ", "_"): name for name in r.fieldnames}
        time_col = field_map.get("timecode")
        cval_col = field_map.get("content_val") or field_map.get("content_value")
        if not (time_col and cval_col):
            raise RuntimeError(f"找不到列名，实际表头为：{r.fieldnames}")

        for row in r:
            tc = (row.get(time_col) or "").strip()
            cv = (row.get(cval_col) or "").strip()
            if not tc or not cv:
                continue
            try:
                times.append(parse_timecode_to_seconds(tc))
                cvals.append(float(cv))
            except ValueError:
                continue

    return np.array(times, dtype=float), np.array(cvals, dtype=float)


def compute_mid_times_for_scenes(
    scenes: List[Tuple],
    t_arr: np.ndarray,
    v_arr: np.ndarray,
    diff_threshold: float
) -> List[Optional[float]]:
    """
    对每个场景，计算“低差异度集合（content_val < diff_threshold）”的中位时间 mid_s。
    若该场景没有低差异度帧，返回 None。
    """
    mid_times: List[Optional[float]] = []
    for (start_tc, end_tc) in scenes:
        start_s = start_tc.get_seconds()
        end_s = end_tc.get_seconds()
        if end_s <= start_s:
            mid_times.append(None)
            continue

        mask_scene = (t_arr >= start_s) & (t_arr < end_s)
        mask_low = mask_scene & (v_arr < diff_threshold)
        if not np.any(mask_low):
            mid_times.append(None)
            continue

        low_times = t_arr[mask_low]
        mid_idx = len(low_times) // 2
        mid_s = float(low_times[mid_idx])
        mid_times.append(mid_s)

    return mid_times


def export_frames_at_times(
    video_path: Path,
    scene_mid_times: List[Optional[float]],
    scenes: List[Tuple],
    out_dir: Path,
    fps_hint_from_video_obj: Optional[object] = None,
    verbose: bool = True,
) -> int:
    """
    将每个场景的 mid_s 帧导出为 PNG。若某场景 mid_s 为 None 则跳过。
    失败时兜底到该场景第一帧。
    返回成功导出的数量。
    """
    if cv2 is None:
        raise RuntimeError("需要安装 opencv-python：pip install opencv-python")

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0 and fps_hint_from_video_obj is not None:
        # PySceneDetect 的 timecode 兜底
        tc = fps_hint_from_video_obj.base_timecode.framerate
        fps = float(tc.num) / float(tc.den)

    exported = 0
    for idx, ((start_tc, end_tc), mid_s) in enumerate(zip(scenes, scene_mid_times), start=1):
        start_s = start_tc.get_seconds()
        if mid_s is None:
            if verbose:
                print(f"[跳过] 场景{idx:04d} 无低差异度中位帧。")
            continue

        mid_frame = int(round(mid_s * fps)) if fps > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            # 兜底：该场景第一帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start_s * fps)) if fps > 0 else 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            if verbose:
                print(f"[跳过] 场景{idx:04d} 读取帧失败。")
            continue

        mid_tc_str = seconds_to_timecode(mid_s)
        out_name = f"scene-{idx:04d}-lowdiff-mid-{mid_tc_str.replace(':','-').replace('.','_')}.png"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), frame)
        exported += 1
        if verbose:
            print(f"[导出] 场景{idx:04d} | 选中 mid={mid_tc_str} -> {out_path.name}")

    cap.release()
    if verbose:
        print(f"完成：共导出 {exported} 张到：{out_dir}/")
    return exported


def plot_content_val(
    video_path: Path,
    scenes: List[Tuple],
    t_arr: np.ndarray,
    v_arr: np.ndarray,
    diff_threshold: float,
    mid_times: List[Optional[float]],
    fig_size: Tuple[float, float] = (16, 9),
    dpi: int = 300,
    save_png: bool = True,
    save_svg: bool = True,
    show: bool = False,
    verbose: bool = True,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    绘制 content_val 曲线 + 场景切点（灰色虚线） + 中位帧红点（在折线上）。
    返回 (png_path, svg_path)，若未保存则为 None。
    """
    plt.figure(figsize=fig_size)

    # 主折线
    plt.plot(t_arr, v_arr, linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("content_val")
    plt.title("PySceneDetect: content_val vs Time")

    # 场景切点（灰色虚线）
    for (start_tc, _end_tc) in scenes:
        cut_sec = start_tc.get_seconds()
        plt.axvline(cut_sec, linestyle="--", alpha=0.5)

    # === 中位帧红点（替代竖线） ===
    # 过滤 None，并构造 x 坐标
    mid_x = [m for m in mid_times if m is not None]
    if mid_x:
        # 用线性插值在 mid_x 处取 y 值，使红点落在折线上
        # 注意：t_arr 需要单调递增（来自 CSV，已满足）
        mid_y = np.interp(mid_x, t_arr, v_arr)
        # 画红色小点：s 点大小，zorder 提升层级
        plt.scatter(mid_x, mid_y, s=16, c="red", zorder=3)

    plt.tight_layout()

    png_path = video_path.with_suffix(".content_val.png") if save_png else None
    svg_path = video_path.with_suffix(".content_val.svg") if save_svg else None

    if save_png:
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
        if verbose:
            print(f"已保存曲线图: {png_path}")
    if save_svg:
        plt.savefig(svg_path, bbox_inches="tight")
        if verbose:
            print(f"已保存矢量图: {svg_path}")

    if show:
        plt.show()

    plt.close()
    return png_path, svg_path


# =========================
# 一键管道（外部最友好）
# =========================
def run_pipeline(cfg: PipelineConfig) -> dict:
    """
    运行完整管道：检测场景 → 导出 CSV → 读取指标 → 计算中位帧 → 导出中位帧 → 绘图。
    返回包含产物路径和统计信息的字典。
    """
    # 1) 场景检测 & CSV
    scenes, csv_out, video_obj = detect_scenes(cfg.video_path, cfg.detect_threshold)
    if cfg.verbose:
        print(f"场景数: {len(scenes)}")
        print(f"已导出逐帧指标: {csv_out}")

    # 2) 读取指标
    t_arr, v_arr = load_stats(csv_out)

    # 3) 计算每个场景的低差异度中位帧时间
    mid_times = compute_mid_times_for_scenes(scenes, t_arr, v_arr, cfg.diff_threshold)

    # 4) 导出中位帧图片
    exported_cnt = 0
    if cfg.out_dir:
        exported_cnt = export_frames_at_times(
            cfg.video_path, mid_times, scenes, cfg.out_dir, fps_hint_from_video_obj=video_obj, verbose=cfg.verbose
        )

    # 5) 绘图
    png_path, svg_path = plot_content_val(
        cfg.video_path, scenes, t_arr, v_arr, cfg.diff_threshold, mid_times,
        fig_size=cfg.fig_size, dpi=cfg.dpi,
        save_png=cfg.save_png, save_svg=cfg.save_svg,
        show=cfg.show_plot, verbose=cfg.verbose
    )

    return {
        "csv": csv_out,
        "png": png_path,
        "svg": svg_path,
        "exported_count": exported_cnt,
        "scenes": scenes,
        "mid_times": mid_times,
        "times": t_arr,
        "cvals": v_arr,
    }


# =========================
# CLI
# =========================
def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="Detect scenes, export mid-frames, and plot content_val.")
    p.add_argument("--video", type=Path, required=True, help="视频文件路径")
    p.add_argument("--detect-th", type=float, default=5.0, help="ContentDetector 阈值")
    p.add_argument("--diff-th", type=float, default=5.0, help="低差异度阈值（content_val < diff_th）")
    p.add_argument("--out-dir", type=Path, default=Path("scene_mids_lowdiff"), help="导出中位帧目录")
    p.add_argument("--fig-w", type=float, default=16.0, help="图宽（英寸）")
    p.add_argument("--fig-h", type=float, default=9.0, help="图高（英寸）")
    p.add_argument("--dpi", type=int, default=300, help="保存 PNG 的 DPI")
    p.add_argument("--no-png", action="store_true", help="不保存 PNG")
    p.add_argument("--no-svg", action="store_true", help="不保存 SVG")
    p.add_argument("--show", action="store_true", help="显示图像窗口（有 GUI 时）")
    p.add_argument("--quiet", action="store_true", help="安静模式（减少日志）")
    return p


def main():
    ap = _build_argparser()
    args = ap.parse_args()

    cfg = PipelineConfig(
        video_path=args.video,
        detect_threshold=args.detect_th,
        diff_threshold=args.diff_th,
        out_dir=args.out_dir,
        fig_size=(args.fig_w, args.fig_h),
        dpi=args.dpi,
        show_plot=args.show,
        save_png=not args.no_png,
        save_svg=not args.no_svg,
        verbose=not args.quiet,
    )

    result = run_pipeline(cfg)
    if cfg.verbose:
        print("完成。概要：")
        print(f"- CSV     : {result['csv']}")
        print(f"- PNG     : {result['png']}")
        print(f"- SVG     : {result['svg']}")
        print(f"- 导出张数: {result['exported_count']}")


if __name__ == "__main__":
    main()
