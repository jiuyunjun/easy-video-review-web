import os
from typing import List, Tuple


def detect_scenes(src: str, threshold: float = 5.0) -> Tuple[float, int, List[Tuple[int, int]]]:
    """Detect scenes using PySceneDetect's ContentDetector.

    Returns a tuple of (fps, total_frame_count, scene_list), where scene_list
    contains (start_frame, end_frame) pairs. Scenes starting within the first
    second of the video are ignored to avoid initial jitter.
    """
    try:
        from scenedetect import VideoManager, SceneManager
        from scenedetect.detectors import ContentDetector
        import cv2
    except Exception:
        return 0.0, 0, []

    video_manager = VideoManager([src])
    scene_manager = SceneManager()
    # threshold is used directly; smaller values generate more cuts
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=10))
    video_manager.set_downscale_factor()
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)
    sd_list = scene_manager.get_scene_list()
    video_manager.release()

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return 0.0, 0, []
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    scene_list = []
    for s, e in sd_list:
        if s.get_seconds() < 1.0:
            continue
        scene_list.append((s.get_frames(), e.get_frames()))
    return fps, frame_count, scene_list


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

    fps, frame_count, scenes = detect_scenes(args.video, args.threshold)
    cap = cv2.VideoCapture(args.video)
    saved = 0
    if not scenes:
        if frame_count > 0:
            mid = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ok, frame = cap.read()
            if ok and frame is not None:
                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if fps > 0 else 0.0
                cv2.imwrite(os.path.join(out_dir, f"scene1_{ts:.3f}.jpg"), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                saved = 1
    else:
        for i, (s, e) in enumerate(scenes, start=1):
            if e <= s:
                continue
            mid = s + (e - s) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if fps > 0 else 0.0
            cv2.imwrite(os.path.join(out_dir, f"scene{i}_{ts:.3f}.jpg"), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved += 1
    cap.release()
    print(f"Saved {saved} snapshots to {out_dir}")
