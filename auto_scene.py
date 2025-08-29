import os
from typing import List, Tuple


def detect_scenes(src: str, threshold: float = 5.0) -> Tuple[float, int, List[Tuple[int, int]], List[int], List[float]]:
    """Detect scenes using PySceneDetect's ContentDetector and collect per-frame metrics.

    Returns (fps, total_frame_count, scene_list, frame_numbers, content_vals) where
    scene_list contains (start_frame, end_frame) pairs (scenes starting within the
    first second are ignored), ``frame_numbers`` is a list of frame indices for
    which metrics were collected, and ``content_vals`` contains the corresponding
    ``content_val`` for each frame.
    """
    try:
        from scenedetect import open_video, SceneManager, StatsManager
        from scenedetect.detectors import ContentDetector
        from scenedetect import FrameTimecode
        import cv2, csv, io
    except Exception:
        return 0.0, 0, [], [], []

    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager=stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video = open_video(src)
    scene_manager.detect_scenes(video=video)
    sd_list = scene_manager.get_scene_list()

    # Extract per-frame metrics from StatsManager using an in-memory CSV.
    csv_buf = io.StringIO()
    stats_manager.save_to_csv(csv_buf)
    csv_buf.seek(0)
    reader = csv.DictReader(csv_buf)
    field_map = {name.strip().lower().replace(" ", "_"): name for name in reader.fieldnames}
    time_col = field_map.get("timecode")
    cval_col = field_map.get("content_val") or field_map.get("content_value")
    frames: List[int] = []
    cvals: List[float] = []
    if time_col and cval_col:
        for row in reader:
            tc = row.get(time_col, "").strip()
            cv = row.get(cval_col, "").strip()
            if not tc or not cv:
                continue
            try:
                ft = FrameTimecode(tc)
                frames.append(ft.get_frames())
                cvals.append(float(cv))
            except Exception:
                continue

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return 0.0, 0, [], frames, cvals
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    scene_list: List[Tuple[int, int]] = []
    for s, e in sd_list:
        if s.get_seconds() < 1.0:
            continue
        scene_list.append((s.get_frames(), e.get_frames()))
    return fps, frame_count, scene_list, frames, cvals


if __name__ == '__main__':
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description='Auto scene detection and frame export.')
    parser.add_argument('video', help='input video file path')
    parser.add_argument('-o', '--out', default=None,
                        help='output folder (default: <video_dir>/.review/<video_stem>)')
    parser.add_argument('-t', '--threshold', type=float, default=5.0,
                        help='scene threshold (default 5, lower detects more cuts)')
    args = parser.parse_args()

    out_dir = args.out
    if not out_dir:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        out_dir = os.path.join(os.path.dirname(args.video), '.review', stem)
    os.makedirs(out_dir, exist_ok=True)

    fps, frame_count, scenes, frames, cvals = detect_scenes(args.video, args.threshold)
    cap = cv2.VideoCapture(args.video)
    saved = 0

    import numpy as np
    f_arr = np.array(frames, dtype=int)
    v_arr = np.array(cvals, dtype=float)

    targets = []
    scene_ranges = scenes if scenes else [(0, frame_count)]
    for s, e in scene_ranges:
        mask = (f_arr >= s) & (f_arr < e) & (v_arr < args.threshold)
        if not np.any(mask):
            continue
        cand = f_arr[mask]
        targets.append(int(cand[len(cand)//2]))

    if not targets and frame_count > 0:
        targets.append(frame_count // 2)

    for i, frame_no in enumerate(targets, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if fps > 0 else 0.0
        cv2.imwrite(os.path.join(out_dir, f"scene{i}_{ts:.3f}.jpg"), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved += 1

    cap.release()
    print(f"Saved {saved} snapshots to {out_dir}")
