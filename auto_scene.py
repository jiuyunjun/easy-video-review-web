import os
import shutil
from typing import List, Tuple


def _try_ffprobe_detect(src: str, threshold: float) -> Tuple[float, int, List[Tuple[int, int]]]:
    """Try detect scenes using ffprobe+lavfi. Returns (fps, frame_count, scene_list)."""
    fps = 0.0
    frame_count = 0
    scene_list: List[Tuple[int, int]] = []
    try:
        import subprocess, json
        if not shutil.which("ffprobe"):
            return fps, frame_count, scene_list

        # Probe stream info
        info_cmd = [
            "ffprobe", "-hide_banner", "-loglevel", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,nb_frames,duration",
            "-of", "json",
            src,
        ]
        info = json.loads(subprocess.check_output(info_cmd).decode("utf-8"))
        stream = info["streams"][0]
        num, den = map(int, stream.get("avg_frame_rate", "0/1").split("/"))
        fps = num / den if den else 0.0
        frame_count = int(stream.get("nb_frames") or 0)
        duration = float(stream.get("duration") or 0.0)
        if frame_count == 0 and fps > 0:
            frame_count = int(duration * fps)

        # Threshold for lavfi select
        ff_t = threshold / 100.0 if threshold > 1 else threshold

        # Use lavfi movie filter. Backslashes/colons/commas must be escaped for lavfi.
        esc = src.replace("\\", "\\\\").replace(":", "\\:").replace(",", "\\,")
        filt = f"movie={esc},select=gt(scene\\,{ff_t})"
        cut_cmd = [
            "ffprobe", "-hide_banner", "-loglevel", "error",
            "-show_frames", "-of", "json", "-f", "lavfi", filt
        ]
        data = json.loads(subprocess.check_output(cut_cmd).decode("utf-8"))
        cuts = [float(f.get("pkt_pts_time", 0)) for f in data.get("frames", [])]
        frame_cuts = [int(t * fps) for t in cuts if t >= 0.2]
        bounds = [0] + frame_cuts + [frame_count]
        scene_list = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
    except Exception:
        return fps, frame_count, []
    return fps, frame_count, scene_list


def _try_pyscenedetect(src: str, threshold: float) -> Tuple[float, int, List[Tuple[int, int]]]:
    """Try PySceneDetect ContentDetector. Returns (fps, frame_count, scene_list)."""
    try:
        from scenedetect import VideoManager, SceneManager
        from scenedetect.detectors import ContentDetector
        import cv2
    except Exception:
        return 0.0, 0, []

    video_manager = VideoManager([src])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.start()
    sd_list = scene_manager.get_scene_list()
    video_manager.release()
    sd_list = [(s, e) for s, e in sd_list if s.get_seconds() >= 1.0]

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return 0.0, 0, []
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    scene_list = [(s.get_frames(), e.get_frames()) for s, e in sd_list]
    return fps, frame_count, scene_list


def _fallback_opencv(src: str, threshold: float) -> Tuple[float, int, List[Tuple[int, int]]]:
    """Pure-OpenCV simple content-change detector as a last resort.
    Uses grayscale frame differences with an adaptive threshold.
    threshold ~ 27 maps to about 0.27 diff fraction.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return 0.0, 0, []

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return 0.0, 0, []
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Pass 1: compute per-sample difference scores
    diffs: List[Tuple[int, float]] = []  # (frame_idx, score)
    prev_small = None
    stride = max(1, int(fps // 2) if fps else 15)  # ~2 samples/sec
    idx = 0
    w_target = 320
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % stride == 0:
            h, w = frame.shape[:2]
            scale = w_target / float(w) if w else 1.0
            small = cv2.resize(frame, (int(w*scale), int(h*scale))) if scale < 1.0 else frame
            small = cv2.GaussianBlur(small, (5,5), 0)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            h_ch, s_ch, v_ch = cv2.split(hsv)
            hist_h = cv2.calcHist([h_ch],[0],None,[32],[0,180])
            hist_s = cv2.calcHist([s_ch],[0],None,[32],[0,256])
            cv2.normalize(hist_h, hist_h)
            cv2.normalize(hist_s, hist_s)
            if prev_small is not None:
                prev_hsv = cv2.cvtColor(prev_small, cv2.COLOR_BGR2HSV)
                ph, ps, pv = cv2.split(prev_hsv)
                ph_h = cv2.calcHist([ph],[0],None,[32],[0,180]); cv2.normalize(ph_h, ph_h)
                ps_h = cv2.calcHist([ps],[0],None,[32],[0,256]); cv2.normalize(ps_h, ps_h)
                # Histogram distance (Bhattacharyya 0..1)
                d_h = float(cv2.compareHist(hist_h, ph_h, cv2.HISTCMP_BHATTACHARYYA))
                d_s = float(cv2.compareHist(hist_s, ps_h, cv2.HISTCMP_BHATTACHARYYA))
                # Luma absdiff
                v_prev = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
                v_now  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                mae = float(cv2.absdiff(v_now, v_prev).mean())/255.0
                score = 0.6*(d_h+d_s)/2.0 + 0.4*mae
                diffs.append((idx, score))
            prev_small = small
        idx += 1
    cap.release()

    if not diffs:
        return fps, frame_count, []

    # Robust threshold: use percentile of scores if user threshold is high
    import numpy as _np
    scores = _np.array([s for _, s in diffs], dtype=_np.float32)
    # Map UI threshold to 0..1 scale; default 0.18
    base_thr = (threshold/100.0 if threshold>1 else threshold) or 0.18
    p85 = float(_np.percentile(scores, 85))
    thr = max(base_thr, p85*0.9)  # adaptive lift

    # Build cuts with minimum spacing (to avoid very short scenes)
    min_len_sec = 1.5
    min_gap = int((fps or 25.0) * min_len_sec)
    cuts: List[int] = []
    last_cut = 0
    for f, s in diffs:
        if s >= thr and (not cuts or f - last_cut >= min_gap):
            cuts.append(f)
            last_cut = f

    if not cuts:
        return fps, frame_count, []
    bounds = [0] + cuts + [frame_count]
    scene_list = [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1) if bounds[i+1]-bounds[i] >= min_gap]
    # if filtering removed all segments, just use coarse bounds
    if not scene_list:
        scene_list = [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]
    return fps, frame_count, scene_list


def detect_scenes(src: str, threshold: float) -> Tuple[float, int, List[Tuple[int, int]]]:
    """Composite detector: ffprobe -> PySceneDetect -> OpenCV fallback."""
    # Try ffprobe with adaptive relaxation if scenes too few
    fps, total, scenes = _try_ffprobe_detect(src, threshold)
    if len(scenes) <= 1 and threshold > 0:
        for fac in (0.75, 0.6, 0.5):
            fps, total, scenes = _try_ffprobe_detect(src, threshold*fac)
            if len(scenes) > 1:
                break
    if scenes:
        return fps, total, scenes
    fps2, total2, scenes2 = _try_pyscenedetect(src, threshold)
    if scenes2:
        return fps2, total2, scenes2
    return _fallback_opencv(src, threshold)


if __name__ == '__main__':
    import argparse
    import cv2
    parser = argparse.ArgumentParser(description='Auto scene detection and mid-frame export.')
    parser.add_argument('video', help='input video file path')
    parser.add_argument('-o', '--out', required=True, help='output folder for snapshots')
    parser.add_argument('-t', '--threshold', type=float, default=18.0, help='scene threshold (lower=more cuts, default 18)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fps, frame_count, scenes = detect_scenes(args.video, args.threshold)
    cap = cv2.VideoCapture(args.video)
    saved = 0
    if not scenes:
        if frame_count > 0:
            mid = int(frame_count // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ok, frame = cap.read()
            if ok and frame is not None:
                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if fps > 0 else 0.0
                cv2.imwrite(os.path.join(args.out, f"scene1_{ts:.3f}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY),95])
                saved = 1
    else:
        for i, (s, e) in enumerate(scenes, start=1):
            if e <= s: continue
            mid = (s + e) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ok, frame = cap.read()
            if not ok or frame is None: continue
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if fps > 0 else 0.0
            cv2.imwrite(os.path.join(args.out, f"scene{i}_{ts:.3f}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY),95])
            saved += 1
    cap.release()
    print(f"Saved {saved} snapshots to {args.out}")
