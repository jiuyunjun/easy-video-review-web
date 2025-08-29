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
        streams = info.get("streams") or []
        if not streams:
            return fps, frame_count, scene_list
        stream = streams[0]
        num, den = map(int, (stream.get("avg_frame_rate") or "0/1").split("/"))
        fps = num / den if den else 0.0
        frame_count = int(stream.get("nb_frames") or 0)
        duration = float(stream.get("duration") or 0.0)
        if frame_count == 0 and fps > 0:
            frame_count = int(duration * fps)

        # Threshold for lavfi select
        ff_t = threshold / 100.0 if threshold > 1 else threshold

        # Use lavfi movie filter. Escape for lavfi.
        esc = src.replace("\\", "\\\\").replace(":", "\\:").replace(",", "\\,")
        # 加一点去抖（decimate），避免高帧率时过密命中
        filt = f"movie={esc},select=gt(scene\\,{ff_t}),metadata=print"
        cut_cmd = [
            "ffprobe", "-hide_banner", "-loglevel", "error",
            "-show_frames", "-of", "json", "-f", "lavfi", filt
        ]
        data = json.loads(subprocess.check_output(cut_cmd).decode("utf-8"))
        frames = data.get("frames", [])
        cuts = [float(f.get("pkt_pts_time", 0)) for f in frames if f.get("pkt_pts_time") is not None]

        # 忽略开头极短抖动
        frame_cuts = [max(0, int(t * fps)) for t in cuts if t >= 0.20]
        # 构段
        bounds = [0] + sorted(set(frame_cuts)) + ([frame_count] if frame_count else [])
        scene_list = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]

        # 合并过近的切点（< 1.0s）
        if fps > 0 and len(scene_list) > 1:
            merged = []
            cur_s, cur_e = scene_list[0]
            min_gap = int(fps * 1.0)
            for s, e in scene_list[1:]:
                if s - cur_e < min_gap:
                    cur_e = e
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            merged.append((cur_s, cur_e))
            scene_list = merged

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

    # 将“用户阈值(0~100或0~1)”映射到 ContentDetector 的范围（通常 15~40）
    t = (threshold if threshold <= 1 else threshold / 100.0)
    t = max(0.0, min(1.0, t))
    sd_thr = 10.0 + 30.0 * t  # 约 10~40，默认 18 相当于 ~15~20

    video_manager = VideoManager([src])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=sd_thr, min_scene_len=10))
    video_manager.set_downscale_factor()  # 自动降采样提速
    video_manager.start()

    # 关键：实际运行检测
    scene_manager.detect_scenes(frame_source=video_manager)
    sd_list = scene_manager.get_scene_list()
    video_manager.release()

    # 过滤掉头一秒内的切点（稳定窗）
    sd_list = [(s, e) for s, e in sd_list if s.get_seconds() >= 1.0]

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return 0.0, 0, []
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    scene_list = [(s.get_frames(), e.get_frames()) for s, e in sd_list]
    return fps, frame_count, scene_list


def _ms_ssim(x, y):
    """简化版 MS-SSIM（避免额外依赖）；若失败回退到普通 SSIM。"""
    try:
        import numpy as np
        import cv2
        # 单尺度 SSIM 近似
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        mu1 = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma1_sq = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu1 * mu1
        sigma2_sq = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu2 * mu2
        sigma12 = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12
        )
        return float(ssim_map.mean())
    except Exception:
        return 0.0


def _fallback_opencv(src: str, threshold: float) -> Tuple[float, int, List[Tuple[int, int]]]:
    """Pure-OpenCV robust detector tuned for mobile screen recordings.

    关键策略：
      - 采样：~3 fps，缩放到宽 512。
      - 掩膜：忽略顶部/底部/左右细边，去除状态栏和导航条干扰。
      - 指标：1-SSIM(灰度) + HSV 直方图巴氏距离 + 边缘差异。
      - 时间平滑：EMA；双阈值滞回；峰值确认；前后稳定窗验证。
      - 防抖：最小场景时长；近切点合并；短段过滤（允许大变化保留）。
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

    # 采样步长：~3 fps
    sample_rate = 8.0  # ~8 samples/sec for screen recordings
    stride = max(1, int(round((fps or 25.0) / sample_rate)))

    # 阈值归一化（用户 0~100 或 0~1）
    t = (threshold if threshold <= 1 else threshold / 100.0)
    t = max(0.0, min(1.0, t))
    # 基础阈值（低阈值更敏感）
    base = 0.16 + 0.20 * t  # 约 0.16~0.36
    hi_thr = base * 1.10
    lo_thr = base * 0.75

    # 时长/距离约束
    min_len_sec = 0.3          # minimum scene length (s)
    confirm_window_sec = 0.18  # confirmation window (s)
    merge_gap_sec = 0.25       # merge nearby cuts (s)
    strong_jump = base * 2.0   # 强变化豁免

    w_target = 512
    idx = 0
    last_kept = None

    scores = []   # (frame_idx, score_raw)
    frames_meta = []  # (idx, ts_msec)

    # 采样遍历
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % stride == 0:
            h, w = frame.shape[:2]
            if w <= 0 or h <= 0:
                idx += 1
                continue
            scale = w_target / float(w) if w > w_target else 1.0
            small = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale)))) if scale < 1.0 else frame

            # 掩膜：去除顶部/底部 8%，左右 3%
            sh, sw = small.shape[:2]
            top = int(sh * 0.08)
            bot = int(sh * 0.92)
            left = int(sw * 0.03)
            right = int(sw * 0.97)
            roi = small[top:bot, left:right]
            if roi.size == 0:
                idx += 1
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            if last_kept is not None:
                prev_roi, prev_gray = last_kept

                # 1) SSIM 距离
                ssim = _ms_ssim(prev_gray.astype('float32'), gray.astype('float32'))
                ssim = max(0.0, min(1.0, ssim))
                d_ssim = 1.0 - ssim

                # 2) HSV 直方图距离（H/S）
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                phsv = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2HSV)
                def _hist(hch, bins, rng):
                    h = cv2.calcHist([hch], [0], None, [bins], rng)
                    return cv2.normalize(h, None).astype('float32')
                d_h = cv2.compareHist(_hist(hsv[...,0], 32, [0,180]),
                                      _hist(phsv[...,0], 32, [0,180]),
                                      cv2.HISTCMP_BHATTACHARYYA)
                d_s = cv2.compareHist(_hist(hsv[...,1], 32, [0,256]),
                                      _hist(phsv[...,1], 32, [0,256]),
                                      cv2.HISTCMP_BHATTACHARYYA)
                d_hs = float(d_h + d_s) * 0.5  # 0..1

                # 3) 边缘图差异（Canny 后 MAE）
                e1 = cv2.Canny(prev_gray, 80, 160)
                e2 = cv2.Canny(gray, 80, 160)
                d_edge = float(cv2.absdiff(e1, e2).mean()) / 255.0  # 0..1

                # 组合：UI 场景变化通常 SSIM 降、色彩块变、边缘结构变
                score = 0.55 * d_ssim + 0.30 * d_hs + 0.15 * d_edge
                scores.append((idx, score))

            last_kept = (roi, gray)
            ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            frames_meta.append((idx, ts))

        idx += 1

    # 用不到帧立即释放
    cap.release()

    if not scores:
        return fps, frame_count, []

    import numpy as np
    # 计算 EMA 平滑
    alpha = 0.35  # 平滑系数
    ema = []
    e = scores[0][1]
    for _, s in scores:
        e = alpha * s + (1 - alpha) * e
        ema.append(e)
    ema = np.array(ema, dtype=np.float32)
    raw = np.array([s for _, s in scores], dtype=np.float32)

    # 自适应上拉：参考 85 分位
    p85 = float(np.percentile(raw, 85))
    lift = max(0.0, p85 * 0.9 - base)
    hi = hi_thr + lift * 0.6
    lo = lo_thr + lift * 0.3

    # 双阈值滞回 + 峰值确认
    min_gap = int(round((fps or 25.0) * merge_gap_sec))
    confirm_w = int(round((fps or 25.0) * confirm_window_sec))
    min_len = int(round((fps or 25.0) * min_len_sec))

    cuts = []
    armed = False
    last_cut = -10**9

    def is_local_peak(i):
        L = max(0, i - confirm_w // stride)
        R = min(len(ema) - 1, i + confirm_w // stride)
        return all(ema[i] >= ema[j] for j in range(L, R + 1))

    for k, (fidx, s) in enumerate(scores):
        val = ema[k]
        if not armed:
            if val >= hi or raw[k] >= strong_jump:
                armed = True
        else:
            if val <= lo and is_local_peak(k):
                if fidx - last_cut >= min_gap:
                    cuts.append(fidx)
                    last_cut = fidx
                armed = False

    # 合并过近切点
    merged = []
    for c in cuts:
        if not merged or c - merged[-1] >= min_gap:
            merged.append(c)
        else:
            # 近切点保留原始分数更高者
            prev = merged[-1]
            pk = scores[[i for i, (fi, _) in enumerate(scores) if fi == prev][0]][1]
            ck = scores[[i for i, (fi, _) in enumerate(scores) if fi == c][0]][1]
            if ck > pk:
                merged[-1] = c
    cuts = merged

    if not cuts:
        return fps, frame_count, []

    # 构段并过滤
    bounds = [0] + cuts + [frame_count if frame_count > 0 else (cuts[-1] + min_len)]
    scene_list = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e <= s:
            continue
        length = e - s
        if length >= min_len:
            scene_list.append((s, e))
        else:
            # 短段只在“强变化”附近保留
            # 找到该切点的原始分数
            if i > 0:
                cut_f = bounds[i]
                # 取最接近 cut_f 的采样索引
                near = min(range(len(scores)), key=lambda j: abs(scores[j][0] - cut_f))
                if raw[near] >= strong_jump:
                    scene_list.append((s, e))
            # 否则丢弃（并由相邻段吞并）

    if not scene_list:
        # 全被过滤时，退回粗分段
        scene_list = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]

    return fps, frame_count, scene_list


def _opencv_stable_segments(src: str, threshold: float) -> Tuple[float, int, List[Tuple[int, int]]]:
    """OpenCV-based detector that returns ONLY stable segments as scenes.

    - Samples ~8 fps with downscale and masked ROI to avoid status/nav bars.
    - Frame-to-frame change score: SSIM distance (gray) + HSV hist Bhattacharyya + edge diff.
    - EMA smoothing + adaptive hysteresis thresholds.
    - Stability definition: EMA below `lo` for a confirm window; end when above `hi` for a confirm window.
    - Merge stable segments separated by very short motion gaps; NO minimum scene length.
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

    sample_rate = 8.0
    stride = max(1, int(round((fps or 25.0) / sample_rate)))

    # Normalize threshold 0..1
    t = (threshold if threshold <= 1 else threshold / 100.0)
    t = max(0.0, min(1.0, t))
    base = 0.16 + 0.20 * t
    hi_thr = base * 1.10
    lo_thr = base * 0.75

    confirm_window_sec = 0.18
    merge_gap_sec = 0.25

    w_target = 512
    idx = 0
    last_kept = None
    scores: List[Tuple[int, float]] = []
    energies: List[float] = []

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % stride == 0:
            h, w = frame.shape[:2]
            if w <= 0 or h <= 0:
                idx += 1
                continue
            scale = w_target / float(w) if w > w_target else 1.0
            small = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale)))) if scale < 1.0 else frame

            sh, sw = small.shape[:2]
            top = int(sh * 0.08); bot = int(sh * 0.92)
            left = int(sw * 0.03); right = int(sw * 0.97)
            roi = small[top:bot, left:right]
            if roi.size == 0:
                idx += 1
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            if last_kept is not None:
                prev_roi, prev_gray = last_kept
                ssim = _ms_ssim(prev_gray.astype('float32'), gray.astype('float32'))
                ssim = max(0.0, min(1.0, ssim))
                d_ssim = 1.0 - ssim

                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                phsv = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2HSV)
                def _hist(hch, bins, rng):
                    h = cv2.calcHist([hch], [0], None, [bins], rng)
                    return cv2.normalize(h, None).astype('float32')
                d_h = cv2.compareHist(_hist(hsv[...,0], 32, [0,180]), _hist(phsv[...,0], 32, [0,180]), cv2.HISTCMP_BHATTACHARYYA)
                d_s = cv2.compareHist(_hist(hsv[...,1], 32, [0,256]), _hist(phsv[...,1], 32, [0,256]), cv2.HISTCMP_BHATTACHARYYA)
                d_hs = float(d_h + d_s) * 0.5

                e1 = cv2.Canny(prev_gray, 80, 160)
                e2 = cv2.Canny(gray, 80, 160)
                d_edge = float(cv2.absdiff(e1, e2).mean()) / 255.0

                score = 0.55 * d_ssim + 0.30 * d_hs + 0.15 * d_edge
                scores.append((idx, score))

                # Optical flow energy (TV-L1 preferred, fallback to Farnebäck)
                try:
                    import numpy as np
                    tvl1 = None
                    if hasattr(cv2, 'optflow') and hasattr(cv2.optflow, 'DualTVL1OpticalFlow_create'):
                        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
                    elif hasattr(cv2, 'DualTVL1OpticalFlow_create'):
                        tvl1 = cv2.DualTVL1OpticalFlow_create()
                    if tvl1 is not None:
                        flow = tvl1.calc(prev_gray, gray, None)
                    else:
                        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                            0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                    # Use robust mean (clip large motions) to reduce noise
                    m = mag.astype('float32')
                    q = np.quantile(m, 0.95)
                    m = np.minimum(m, q)
                    e_val = float(m.mean())
                except Exception:
                    e_val = float(abs(float(d_ssim)))  # fallback proxy
                energies.append(e_val)

            last_kept = (roi, gray)
        idx += 1

    cap.release()

    if not scores:
        return fps, frame_count, []

    import numpy as np
    alpha = 0.35
    ema = []
    e = scores[0][1]
    for _, s in scores:
        e = alpha * s + (1 - alpha) * e
        ema.append(e)
    ema = np.array(ema, dtype=np.float32)
    raw = np.array([s for _, s in scores], dtype=np.float32)

    # Energy EMA
    if energies:
        eema = []
        ee = float(energies[0])
        for v in energies:
            ee = 0.5 * v + 0.5 * ee
            eema.append(ee)
        eema = np.array(eema, dtype=np.float32)
    else:
        eema = np.zeros(len(ema), dtype=np.float32)

    p85 = float(np.percentile(raw, 85))
    lift = max(0.0, p85 * 0.9 - base)
    hi = hi_thr + lift * 0.6
    lo = lo_thr + lift * 0.3

    # Adaptive thresholds for energy hysteresis
    if len(eema) > 0:
        p20 = float(np.percentile(eema, 20))
        p40 = float(np.percentile(eema, 40))
        lo_e = p20
        hi_e = max(p20 * 1.3, p40 * 1.10)
    else:
        lo_e = 0.0
        hi_e = 0.0

    confirm_w = int(round((fps or 25.0) * confirm_window_sec))
    merge_gap_frames = int(round((fps or 25.0) * merge_gap_sec))
    confirm_n = max(1, confirm_w // max(1, stride))
    merge_gap_n = max(0, merge_gap_frames // max(1, stride))

    # Stable only when both appearance and motion are stable
    stable_flags = (ema <= lo) & (eema <= lo_e)

    scenes: List[Tuple[int, int]] = []
    i = 0
    N = len(ema)
    while i < N:
        if not stable_flags[i]:
            i += 1
            continue
        run_start = i
        while i < N and stable_flags[i]:
            i += 1
        run_end = i
        if run_end - run_start < confirm_n:
            continue
        j = run_end
        while j < N:
            gap_start = j
            while j < N and not stable_flags[j]:
                j += 1
            gap_len = j - gap_start
            if gap_len == 0 or gap_len > merge_gap_n:
                break
            next_stable_start = j
            while j < N and stable_flags[j]:
                j += 1
            next_stable_end = j
            if next_stable_end - next_stable_start >= confirm_n:
                run_end = next_stable_end
            else:
                break
        start_frame = scores[run_start][0]
        last_idx = scores[min(run_end - 1, len(scores) - 1)][0]
        end_frame = min(frame_count, last_idx + stride)
        if end_frame > start_frame:
            scenes.append((start_frame, end_frame))
    return fps, frame_count, scenes
def detect_scenes(src: str, threshold: float) -> Tuple[float, int, List[Tuple[int, int]]]:
    """Detect scenes as STABLE segments (no minimum length)."""
    return _opencv_stable_segments(src, threshold)


if __name__ == '__main__':
    import argparse
    import cv2
    parser = argparse.ArgumentParser(description='Auto scene detection and one-third-frame export.')
    parser.add_argument('video', help='input video file path')
    parser.add_argument('-o', '--out', required=False, default=None,
                        help='output folder (default: <video_dir>/.review/<video_stem>)')
    parser.add_argument('-t', '--threshold', type=float, default=18.0, help='scene threshold (lower=more cuts, default 18)')
    args = parser.parse_args()

    # Default output folder if not provided
    out_dir = args.out
    if not out_dir:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        out_dir = os.path.join(os.path.dirname(args.video), '.review', stem)
    os.makedirs(out_dir, exist_ok=True)
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
                cv2.imwrite(os.path.join(out_dir, f"scene1_{ts:.3f}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY),95])
                saved = 1
    else:
        for i, (s, e) in enumerate(scenes, start=1):
            if e <= s: continue
            one_third = s + (e - s) // 3
            cap.set(cv2.CAP_PROP_POS_FRAMES, one_third)
            ok, frame = cap.read()
            if not ok or frame is None: continue
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if fps > 0 else 0.0
            cv2.imwrite(os.path.join(out_dir, f"scene{i}_{ts:.3f}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY),95])
            saved += 1
    cap.release()
    print(f"Saved {saved} snapshots to {out_dir}")
