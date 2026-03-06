"""
Optical-flow-based highlight / high-motion detection.

Uses the Farneback dense optical-flow algorithm to measure frame-to-frame
motion intensity and returns timestamps where motion exceeds a threshold.
Only processes keyframes (I-frames) for speed.
"""

import cv2
import re
import numpy as np
import subprocess


def _get_keyframe_timestamps(video_path: str) -> list[float]:
    """Extract keyframe timestamps using ffmpeg showinfo filter."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"

    cmd = [
        ffmpeg, "-i", video_path,
        "-vf", "select=eq(pict_type\\,I),showinfo",
        "-vsync", "vfr",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse pts_time from showinfo output on stderr
    timestamps = []
    for match in re.finditer(r"pts_time:\s*([\d.]+)", result.stderr):
        timestamps.append(float(match.group(1)))
    return timestamps


def detect_highlights(video_path: str, threshold: float = 2.0) -> list[dict]:
    """
    Analyse a video file for high-motion highlights using dense optical flow.
    Only processes keyframes (I-frames) for fast analysis on long videos.

    Args:
        video_path: Path to the video file.
        threshold: Motion intensity threshold. Lower = more sensitive.

    Returns:
        A list of dicts, each with:
            - ``timestamp`` (float): time in seconds
            - ``intensity`` (float): mean optical-flow magnitude at that point
    """
    keyframe_ts = _get_keyframe_timestamps(video_path)
    if len(keyframe_ts) < 2:
        return []

    cap = cv2.VideoCapture(video_path)
    highlights: list[dict] = []
    prev_gray = None

    for ts in keyframe_ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Calculate Dense Optical Flow (Farneback Algorithm)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Convert Cartesian to Polar (magnitude, angle)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Calculate the mean motion intensity of the frame
            avg_motion = float(np.mean(mag))

            if avg_motion > threshold:
                highlights.append({
                    "timestamp": round(ts, 2),
                    "intensity": round(avg_motion, 2),
                })

        prev_gray = gray

    cap.release()
    return highlights
