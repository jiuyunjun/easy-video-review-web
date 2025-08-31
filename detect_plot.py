from __future__ import annotations

"""Utilities for scene detection plotting.

This module provides a pipeline that runs PySceneDetect to detect scenes,
exports optional mid-frames, and generates a plot of the `content_val`
curve with scene cuts and selected mid points. The plotting logic is
adapted from the reference implementation provided by the user.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import os
import math
import csv

# Switch backend when no display is available (e.g. in headless envs).
import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from scenedetect import open_video, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector


def parse_timecode_to_seconds(tc: str) -> float:
    """Parse HH:MM:SS(.ms) style timecodes to seconds."""
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
    """Return HH:MM:SS.mmm style string."""
    if sec < 0:
        sec = 0.0
    ms = int(round((sec - math.floor(sec)) * 1000))
    s = int(sec) % 60
    m = (int(sec) // 60) % 60
    h = int(sec) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@dataclass
class PipelineConfig:
    video_path: Path
    detect_threshold: float = 5.0
    diff_threshold: float = 5.0
    out_dir: Optional[Path] = Path("scene_mids_lowdiff")
    work_dir: Optional[Path] = None
    fig_size: Tuple[float, float] = (16.0, 9.0)
    dpi: int = 300
    show_plot: bool = False
    save_png: bool = True
    save_svg: bool = True
    verbose: bool = True


def detect_scenes(video_path: Path, detect_threshold: float, work_dir: Path) -> Tuple[List[Tuple], Path, object]:
    """Run PySceneDetect and export a CSV of metrics."""
    assert video_path.exists(), f"video not found: {video_path!r}"
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager=stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=detect_threshold))
    video = open_video(str(video_path))
    scene_manager.detect_scenes(video=video)
    scenes = scene_manager.get_scene_list()
    work_dir.mkdir(parents=True, exist_ok=True)
    csv_out = work_dir / f"{video_path.stem}.stats.csv"
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        scene_manager.stats_manager.save_to_csv(f)
    return scenes, csv_out, video


def load_stats(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    times: List[float] = []
    cvals: List[float] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise RuntimeError(f"CSV missing header: {csv_path}")
        field_map = {name.strip().lower().replace(" ", "_"): name for name in r.fieldnames}
        time_col = field_map.get("timecode")
        cval_col = field_map.get("content_val") or field_map.get("content_value")
        if not (time_col and cval_col):
            raise RuntimeError(f"unexpected columns: {r.fieldnames}")
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
    if cv2 is None:
        raise RuntimeError("opencv-python is required to export frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0 and fps_hint_from_video_obj is not None:
        tc = fps_hint_from_video_obj.base_timecode.framerate
        fps = float(tc.num) / float(tc.den)
    exported = 0
    for idx, ((start_tc, end_tc), mid_s) in enumerate(zip(scenes, scene_mid_times), start=1):
        start_s = start_tc.get_seconds()
        if mid_s is None:
            if verbose:
                print(f"[skip] scene{idx:04d} no mid frame")
            continue
        mid_frame = int(round(mid_s * fps)) if fps > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start_s * fps)) if fps > 0 else 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            if verbose:
                print(f"[skip] scene{idx:04d} read fail")
            continue
        mid_tc_str = seconds_to_timecode(mid_s)
        out_name = f"scene-{idx:04d}-lowdiff-mid-{mid_tc_str.replace(':','-').replace('.','_')}.png"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), frame)
        exported += 1
        if verbose:
            print(f"[export] scene{idx:04d} -> {out_path.name}")
    cap.release()
    if verbose:
        print(f"exported {exported} frames to {out_dir}/")
    return exported


def plot_content_val(
    video_path: Path,
    scenes: List[Tuple],
    t_arr: np.ndarray,
    v_arr: np.ndarray,
    diff_threshold: float,
    mid_times: List[Optional[float]],
    work_dir: Path,
    fig_size: Tuple[float, float] = (16, 9),
    dpi: int = 300,
    save_png: bool = True,
    save_svg: bool = True,
    show: bool = False,
    verbose: bool = True,
) -> Tuple[Optional[Path], Optional[Path]]:
    plt.figure(figsize=fig_size)
    plt.plot(t_arr, v_arr, linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("content_val")
    plt.title("PySceneDetect: content_val vs Time")
    # thin horizontal reference lines every 5 units on the y-axis
    y_max = float(v_arr.max()) if v_arr.size else 0.0
    y_max = math.ceil(y_max / 5.0) * 5.0
    for y in np.arange(0, y_max + 0.1, 5.0):
        plt.axhline(y, color="gray", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    for (start_tc, _end_tc) in scenes:
        cut_sec = start_tc.get_seconds()
        plt.axvline(cut_sec, linestyle="--", alpha=0.5)
    mid_x = [m for m in mid_times if m is not None]
    if mid_x:
        mid_y = np.interp(mid_x, t_arr, v_arr)
        plt.scatter(mid_x, mid_y, s=16, c="red", zorder=3)
    plt.tight_layout()
    work_dir.mkdir(parents=True, exist_ok=True)
    png_path = work_dir / f"{video_path.stem}.content_val.png" if save_png else None
    svg_path = work_dir / f"{video_path.stem}.content_val.svg" if save_svg else None
    if save_png:
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
        if verbose:
            print(f"plot saved: {png_path}")
    if save_svg:
        plt.savefig(svg_path, bbox_inches="tight")
        if verbose:
            print(f"plot saved: {svg_path}")
    if show:
        plt.show()
    plt.close()
    return png_path, svg_path


def run_pipeline(cfg: PipelineConfig) -> dict:
    work_dir = cfg.work_dir or cfg.video_path.parent
    scenes, csv_out, video_obj = detect_scenes(cfg.video_path, cfg.detect_threshold, work_dir)
    if cfg.verbose:
        print(f"scenes: {len(scenes)}")
        print(f"stats: {csv_out}")
    t_arr, v_arr = load_stats(csv_out)
    mid_times = compute_mid_times_for_scenes(scenes, t_arr, v_arr, cfg.diff_threshold)
    exported_cnt = 0
    if cfg.out_dir:
        exported_cnt = export_frames_at_times(
            cfg.video_path,
            mid_times,
            scenes,
            cfg.out_dir,
            fps_hint_from_video_obj=video_obj,
            verbose=cfg.verbose,
        )
    png_path, svg_path = plot_content_val(
        cfg.video_path,
        scenes,
        t_arr,
        v_arr,
        cfg.diff_threshold,
        mid_times,
        work_dir,
        fig_size=cfg.fig_size,
        dpi=cfg.dpi,
        save_png=cfg.save_png,
        save_svg=cfg.save_svg,
        show=cfg.show_plot,
        verbose=cfg.verbose,
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


__all__ = [
    "PipelineConfig",
    "run_pipeline",
    "plot_content_val",
    "detect_scenes",
]
