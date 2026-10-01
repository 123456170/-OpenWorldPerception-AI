import os
import json
import time
import math
import tempfile
from collections import deque, defaultdict
from datetime import datetime, timezone

import numpy as np
import cv2
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# OpenWorldPerception AI
# Mission: Detect known objects while discovering and clustering
# previously unseen object categories.
# ------------------------------------------------------------

st.set_page_config(
    page_title="OpenWorldPerception AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌍 OpenWorldPerception AI")
st.caption(
    "Open-world recognition prototype: known-object detection + temporal tracking + "
    "incremental discovery/clustering of unseen categories. No API key required."
)

KNOWN_CLASSES = {
    "safety_cone": {
        "lower": (5, 120, 120),
        "upper": (25, 255, 255),
        "color_bgr": (0, 140, 255),
    },
    "service_robot": {
        "lower": (95, 110, 80),
        "upper": (130, 255, 255),
        "color_bgr": (255, 80, 0),
    },
    "package": {
        "lower": (16, 90, 120),
        "upper": (35, 255, 255),
        "color_bgr": (0, 200, 255),
    },
}


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def clamp(value, low, high):
    return max(low, min(high, value))


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih

    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def bbox_center(box):
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def nms(detections, iou_threshold=0.45):
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)

        remaining = []
        for det in detections:
            if bbox_iou(best["bbox"], det["bbox"]) < iou_threshold:
                remaining.append(det)
        detections = remaining

    return keep


def draw_label(img, text, org, color=(255, 255, 255)):
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1

    try:
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    except Exception:
        tw, th, baseline = 80, 12, 2

    x = int(max(0, x))
    y = int(max(th + 6, y))

    cv2.rectangle(
        img,
        (x - 2, y - th - 4),
        (x + tw + 2, y + baseline + 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(img, text, (x, y - 2), font, scale, color, thickness, cv2.LINE_AA)


# ------------------------------------------------------------
# Synthetic deterministic demo source
# ------------------------------------------------------------

class SyntheticDemoSource:
    """
    Deterministic synthetic warehouse/street style scene.

    Known objects:
      - safety_cone
      - service_robot
      - package

    Previously unseen / novel objects:
      - unseen_gadget
      - unseen_container
      - anomalous_spill
    """

    def __init__(self, width=640, height=360, fps=12, seed=42):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.frame_idx = 0
        self.rng = np.random.default_rng(seed)
        self.background = self._make_background()

    def _make_background(self):
        base = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        top = np.array([55, 35, 25], dtype=np.float32)
        bottom = np.array([25, 25, 45], dtype=np.float32)

        for y in range(self.height):
            alpha = y / max(1, self.height - 1)
            color = (1.0 - alpha) * top + alpha * bottom
            base[y, :] = color.astype(np.uint8)

        horizon = int(self.height * 0.45)

        for x in range(0, self.width, max(20, self.width // 16)):
            cv2.line(base, (x, horizon), (self.width // 2, self.height), (50, 50, 70), 1)
            cv2.line(base, (self.width - x, horizon), (self.width // 2, self.height), (50, 50, 70), 1)

        cv2.line(base, (0, horizon), (self.width, horizon), (70, 70, 95), 2)

        return base

    def read(self):
        t = self.frame_idx / max(1e-6, self.fps)
        img = self.background.copy()

        light = 0.82 + 0.18 * math.sin(0.35 * t)
        img = np.clip(img.astype(np.float32) * light, 0, 255).astype(np.uint8)

        gt = []

        # Known object 1: safety cone
        cx = int(self.width * (0.18 + 0.14 * math.sin(0.65 * t)))
        cy = int(self.height * (0.70 + 0.04 * math.sin(1.15 * t)))
        s = int(max(14, self.width * 0.032 + 3 * math.sin(1.7 * t)))

        pts = np.array(
            [
                [cx, cy + s],
                [cx + s, cy + s],
                [cx + s // 2, cy - s],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(img, [pts], (0, 140, 255))
        cv2.polylines(img, [pts], True, (0, 80, 200), 2)
        cone_box = cv2.boundingRect(pts)
        gt.append(
            {
                "label": "safety_cone",
                "known": True,
                "bbox": tuple(int(v) for v in cone_box),
            }
        )

        # Known object 2: service robot
        rx = int(self.width * (0.62 + 0.18 * math.sin(0.42 * t + 1.2)))
        ry = int(self.height * (0.52 + 0.06 * math.cos(0.57 * t)))
        rw = int(self.width * 0.075)
        rh = int(self.height * 0.12)

        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (255, 80, 0), -1)
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (180, 50, 0), 2)
        cv2.circle(img, (rx + rw // 2, ry + rh // 3), max(3, rw // 8), (255, 255, 255), -1)
        gt.append(
            {
                "label": "service_robot",
                "known": True,
                "bbox": (rx, ry, rw, rh),
            }
        )

        # Known object 3: package
        px = int(self.width * (0.42 + 0.04 * math.sin(0.22 * t + 2.0)))
        py = int(self.height * 0.76)
        ps = int(self.width * 0.065)

        cv2.rectangle(img, (px, py), (px + ps, py + int(ps * 0.75)), (0, 200, 255), -1)
        cv2.rectangle(img, (px, py), (px + ps, py + int(ps * 0.75)), (0, 120, 200), 2)
        gt.append(
            {
                "label": "package",
                "known": True,
                "bbox": (px, py, ps, int(ps * 0.75)),
            }
        )

        # Unknown object 1: unseen gadget
        if t > 2.5:
            gx = int(self.width * (0.82 - 0.10 * math.sin(0.75 * t)))
            gy = int(self.height * (0.32 + 0.05 * math.sin(0.9 * t)))
            ax = int(self.width * 0.035)
            ay = int(self.height * 0.050)

            cv2.ellipse(img, (gx, gy), (ax, ay), 0, 0, 360, (255, 0, 255), -1)
            cv2.ellipse(img, (gx, gy), (ax, ay), 0, 0, 360, (180, 0, 180), 2)
            gt.append(
                {
                    "label": "unseen_gadget",
                    "known": False,
                    "bbox": (gx - ax, gy - ay, 2 * ax, 2 * ay),
                }
            )

        # Unknown object 2: unseen container, blinking to test temporal association
        if t > 6.0 and int(t * 1.5) % 3 != 0:
            ux = int(self.width * (0.28 + 0.08 * math.cos(0.5 * t)))
            uy = int(self.height * (0.38 + 0.04 * math.sin(0.8 * t + 1.0)))
            us = int(self.width * 0.05)

            cv2.rectangle(img, (ux, uy), (ux + us, uy + us), (200, 0, 120), -1)
            cv2.rectangle(img, (ux, uy), (ux + us, uy + us), (120, 0, 80), 2)
            gt.append(
                {
                    "label": "unseen_container",
                    "known": False,
                    "bbox": (ux, uy, us, us),
                }
            )

        # Unknown object 3: anomalous spill
        if t > 9.0:
            sx = int(self.width * (0.55 + 0.07 * math.sin(0.35 * t + 2.0)))
            sy = int(self.height * 0.84)
            spill_pts = np.array(
                [
                    [sx - 34, sy],
                    [sx - 12, sy - 18],
                    [sx + 16, sy - 12],
                    [sx + 38, sy + 2],
                    [sx + 10, sy + 16],
                    [sx - 18, sy + 12],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(img, [spill_pts], (0, 255, 0))
            cv2.polylines(img, [spill_pts], True, (0, 160, 0), 2)
            spill_box = cv2.boundingRect(spill_pts)
            gt.append(
                {
                    "label": "anomalous_spill",
                    "known": False,
                    "bbox": tuple(int(v) for v in spill_box),
                }
            )

        # Moving occluder to test temporal robustness
        if t > 12.0:
            ox = int(self.width * ((0.05 * t) % 1.2) - 0.1 * self.width)
            cv2.rectangle(img, (ox, 0), (ox + 30, self.height // 2), (90, 90, 90), -1)

        # Sensor noise
        noise = self.rng.normal(0.0, 2.5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        frame = img
        self.frame_idx += 1

        return frame, gt, t


class FileVideoSource:
    def __init__(self, path, width=640, height=360, fps=12):
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.frame_idx = 0

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {path}")

        file_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not file_fps or math.isnan(file_fps) or file_fps <= 0:
            file_fps = fps
        self.fps = float(file_fps)

    def read(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None, None

        h, w = frame.shape[:2]
        if w > self.width:
            new_h = int(h * self.width / w)
            frame = cv2.resize(frame, (self.width, new_h), interpolation=cv2.INTER_AREA)

        ts = self.frame_idx / max(1e-6, self.fps)
        self.frame_idx += 1
        return frame, [], ts


class WebcamSource:
    def __init__(self, width=640, height=360, index=0):
        self.width = int(width)
        self.height = int(height)
        self.frame_idx = 0
        self.start_time = time.time()

        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise IOError("Webcam is unavailable or permission was denied.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise IOError("Webcam frame capture failed.")

        ts = time.time() - self.start_time
        self.frame_idx += 1
        return frame, [], ts


# ------------------------------------------------------------
# Perception components
# ------------------------------------------------------------

def detect_known(frame, config):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_area = frame.shape[0] * frame.shape[1]

    min_area = max(100, int(frame_area * 0.00045))
    max_area = int(frame_area * 0.35)
    ideal_area = max(500, int(frame_area * 0.006))

    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    detections = []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for label, class_cfg in KNOWN_CLASSES.items():
        lower = np.array(class_cfg["lower"], dtype=np.uint8)
        upper = np.array(class_cfg["upper"], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        combined_mask = cv2.bitwise_or(combined_mask, mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        class_dets = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 5 or h < 5:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue

            solidity = area / hull_area
            extent = area / float(w * h + 1e-6)
            aspect = w / float(h + 1e-6)

            area_score = clamp(
                1.0 - abs(math.log(area + 1.0) - math.log(ideal_area + 1.0)) / 2.5,
                0.0,
                1.0,
            )

            roi_mask = mask[y : y + h, x : x + w]
            color_score = clamp(float(roi_mask.mean()) / 255.0, 0.0, 1.0) if roi_mask.size else 0.0
            shape_score = clamp(0.6 * solidity + 0.4 * extent, 0.0, 1.0)

            confidence = 0.45 * color_score + 0.35 * area_score + 0.20 * shape_score

            if confidence < float(config.get("min_candidate_conf", 0.12)):
                continue

            class_dets.append(
                {
                    "label": label,
                    "known": True,
                    "confidence": float(confidence),
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "measurements": {
                        "area_px": float(area),
                        "solidity": float(solidity),
                        "extent": float(extent),
                        "aspect": float(aspect),
                    },
                }
            )

        class_dets = nms(class_dets, iou_threshold=0.45)
        detections.extend(class_dets)

    return detections, combined_mask


def detect_unknown(frame, known_mask, bg_subtractor, config):
    frame_area = frame.shape[0] * frame.shape[1]

    fg_mask = bg_subtractor.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat_mask = cv2.inRange(hsv, np.array([0, 100, 90]), np.array([179, 255, 255]))

    unknown_mask = cv2.bitwise_or(fg_mask, sat_mask)
    unknown_mask = cv2.bitwise_and(unknown_mask, cv2.bitwise_not(known_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    unknown_mask = cv2.morphologyEx(unknown_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    unknown_mask = cv2.morphologyEx(unknown_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(unknown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(80, int(frame_area * 0.00035))
    max_area = int(frame_area * 0.35)

    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue

        roi = unknown_mask[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        mask_score = clamp(float(roi.mean()) / 255.0, 0.0, 1.0)
        area_score = clamp(
            math.log1p(area) / math.log1p(max(1000.0, frame_area * 0.01)),
            0.0,
            1.0,
        )

        confidence = clamp(0.68 * mask_score + 0.32 * area_score, 0.0, 1.0)

        if confidence < float(config.get("min_candidate_conf", 0.12)):
            continue

        detections.append(
            {
                "label": "unclassified",
                "known": False,
                "confidence": float(confidence),
                "bbox": (int(x), int(y), int(w), int(h)),
                "measurements": {
                    "area_px": float(area),
                    "aspect": float(w / (h + 1e-6)),
                    "mask_score": float(mask_score),
                    "salience": "background/color",
                },
            }
        )

    detections = nms(detections, iou_threshold=0.35)
    return detections[: int(config.get("max_detections", 20))]


def compute_embedding(frame, bbox):
    H, W = frame.shape[:2]
    x, y, w, h = [int(v) for v in bbox]

    pad_x = int(w * 0.15)
    pad_y = int(h * 0.15)

    x0 = clamp(x - pad_x, 0, W - 1)
    y0 = clamp(y - pad_y, 0, H - 1)
    x1 = clamp(x + w + pad_x, 1, W)
    y1 = clamp(y + h + pad_y, 1, H)

    crop = frame[y0:y1, x0:x1]

    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return np.zeros(216, dtype=np.float32)

    small = cv2.resize(crop, (8, 8), interpolation=cv2.INTER_AREA)
    pixels = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32).flatten() / 255.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [4], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [4], [0, 256]).flatten()

    hist_h = hist_h / (np.linalg.norm(hist_h) + 1e-6)
    hist_s = hist_s / (np.linalg.norm(hist_s) + 1e-6)
    hist_v = hist_v / (np.linalg.norm(hist_v) + 1e-6)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edge = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    edge_abs = np.abs(edge)
    edge_feats = np.array(
        [
            float(edge_abs.mean()) / 255.0,
            float(edge_abs.std()) / 255.0,
        ],
        dtype=np.float32,
    )

    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
    meanstd = np.concatenate(
        [
            rgb_small.mean(axis=(0, 1)).flatten() / 255.0,
            rgb_small.std(axis=(0, 1)).flatten() / 255.0,
        ]
    ).astype(np.float32)

    feats = np.concatenate([pixels, hist_h, hist_s, hist_v, edge_feats, meanstd])
    feats = feats / (np.linalg.norm(feats) + 1e-6)

    return feats.astype(np.float32)


class IncrementalDiscovery:
    """
    Incremental nearest-centroid novelty clustering.

    Purpose:
      - Assign unknown detections to an existing discovered category if similar.
      - Create a new category candidate when the observation is novel.
    """

    def __init__(self, similarity_threshold=0.74, max_clusters=24):
        self.similarity_threshold = float(similarity_threshold)
        self.max_clusters = int(max_clusters)
        self.clusters = {}
        self.next_id = 1

    def assign(self, embedding):
        if embedding is None:
            embedding = np.zeros(216, dtype=np.float32)

        norm = np.linalg.norm(embedding)
        if norm < 1e-6:
            embedding = np.ones_like(embedding, dtype=np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

        if not self.clusters:
            cluster_id = self.next_id
            self.next_id += 1
            self.clusters[cluster_id] = {
                "id": cluster_id,
                "label": f"novel_category_{cluster_id:02d}",
                "centroid": embedding.copy(),
                "count": 1,
                "created_utc": utc_now_iso(),
            }
            return cluster_id, 1.0, True

        best_id = None
        best_sim = -1.0

        for cluster_id, cluster in self.clusters.items():
            centroid = cluster["centroid"]

            if centroid.shape != embedding.shape:
                continue

            sim = float(np.dot(embedding, centroid))
            if sim > best_sim:
                best_sim = sim
                best_id = cluster_id

        novelty = clamp(1.0 - best_sim, 0.0, 1.0)

        if best_id is None:
            best_id = int(list(self.clusters.keys())[0])
            best_sim = 0.0

        if best_sim >= self.similarity_threshold or len(self.clusters) >= self.max_clusters:
            cluster = self.clusters[best_id]
            new_centroid = (cluster["centroid"] * cluster["count"] + embedding) / (cluster["count"] + 1)
            cluster["centroid"] = new_centroid / (np.linalg.norm(new_centroid) + 1e-6)
            cluster["count"] += 1
            return best_id, novelty, False

        cluster_id = self.next_id
        self.next_id += 1
        self.clusters[cluster_id] = {
            "id": cluster_id,
            "label": f"novel_category_{cluster_id:02d}",
            "centroid": embedding.copy(),
            "count": 1,
            "created_utc": utc_now_iso(),
        }

        return cluster_id, novelty, True


class SimpleTracker:
    """
    Lightweight IoU/centroid tracker for temporal consistency.
    """

    def __init__(self, max_missed=10, iou_threshold=0.18, max_dist_frac=0.18):
        self.tracks = {}
        self.next_id = 1
        self.max_missed = int(max_missed)
        self.iou_threshold = float(iou_threshold)
        self.max_dist_frac = float(max_dist_frac)

    def update(self, detections, width, height, dt):
        active = [(tid, tr) for tid, tr in self.tracks.items() if tr["missed"] <= self.max_missed]
        pairs = []

        for det_idx, det in enumerate(detections):
            dc = bbox_center(det["bbox"])

            for track_id, track in active:
                iou = bbox_iou(det["bbox"], track["bbox"])
                tc = bbox_center(track["bbox"])
                dist = math.hypot(dc[0] - tc[0], dc[1] - tc[1]) / float(max(1, max(width, height)))

                if iou >= self.iou_threshold or dist <= self.max_dist_frac:
                    score = iou + (1.0 - min(1.0, dist)) * 0.45

                    if det["known"] == track["known"] and det["label"] == track["label"]:
                        score += 0.15

                    pairs.append((score, det_idx, track_id))

        pairs.sort(key=lambda x: x[0], reverse=True)

        matched_det = set()
        matched_track = set()

        for score, det_idx, track_id in pairs:
            if det_idx in matched_det or track_id in matched_track:
                continue

            det = detections[det_idx]
            track = self.tracks[track_id]

            prev_c = bbox_center(track["bbox"])
            curr_c = bbox_center(det["bbox"])

            vx = (curr_c[0] - prev_c[0]) / max(dt, 1e-3)
            vy = (curr_c[1] - prev_c[1]) / max(dt, 1e-3)

            track["bbox"] = det["bbox"]
            track["label"] = det["label"]
            track["known"] = det["known"]
            track["missed"] = 0
            track["hits"] += 1
            track["age"] += 1
            track["conf"] = 0.7 * track["conf"] + 0.3 * float(det["confidence"])

            if not det["known"] and det.get("cluster_id") is not None:
                cluster_id = det["cluster_id"]
                track["cluster_votes"][cluster_id] = track["cluster_votes"].get(cluster_id, 0) + 1
                track["cluster_id"] = cluster_id

            det["track_id"] = track_id
            det["velocity_px_s"] = (float(vx), float(vy))

            if "measurements" in det:
                det["measurements"]["velocity_px_s"] = round(math.hypot(vx, vy), 2)

            matched_det.add(det_idx)
            matched_track.add(track_id)

        for det_idx, det in enumerate(detections):
            if det_idx in matched_det:
                continue

            if float(det["confidence"]) < 0.18:
                continue

            track_id = self.next_id
            self.next_id += 1

            cluster_votes = {}
            if not det["known"] and det.get("cluster_id") is not None:
                cluster_votes[det["cluster_id"]] = 1

            self.tracks[track_id] = {
                "bbox": det["bbox"],
                "label": det["label"],
                "known": det["known"],
                "conf": float(det["confidence"]),
                "hits": 1,
                "missed": 0,
                "age": 1,
                "cluster_id": det.get("cluster_id"),
                "cluster_votes": cluster_votes,
                "created_utc": utc_now_iso(),
            }

            det["track_id"] = track_id
            det["velocity_px_s"] = (0.0, 0.0)

            matched_det.add(det_idx)
            matched_track.add(track_id)

        for track_id, track in self.tracks.items():
            if track_id not in matched_track:
                track["missed"] += 1

        self.tracks = {
            tid: tr for tid, tr in self.tracks.items() if tr["missed"] <= self.max_missed
        }

        return detections


class EvaluationAccumulator:
    """
    Evaluation metrics for the demo scene:
      - known-object precision/recall/F1
      - mean IoU for true positives
      - calibration error
      - ID switches
      - unseen-category clustering accuracy/purity
    """

    def __init__(self):
        self.frames = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.iou_sum = 0.0
        self.calibration = []
        self.id_switches = 0
        self.gt_track_map = {}
        self.unknown_votes = defaultdict(lambda: defaultdict(int))
        self.unknown_matched = 0

    def update(self, detections, gt):
        self.frames += 1

        known_gt = [g for g in gt if g.get("known", False)]
        known_dets = [d for d in detections if d.get("known", False)]

        pairs = []

        for det_idx, det in enumerate(known_dets):
            for gt_idx, g in enumerate(known_gt):
                if det["label"] != g["label"]:
                    continue

                iou = bbox_iou(det["bbox"], g["bbox"])
                if iou >= 0.30:
                    pairs.append((iou, det_idx, gt_idx))

        pairs.sort(key=lambda x: x[0], reverse=True)

        matched_det = set()
        matched_gt = set()

        for iou, det_idx, gt_idx in pairs:
            if det_idx in matched_det or gt_idx in matched_gt:
                continue

            matched_det.add(det_idx)
            matched_gt.add(gt_idx)

            self.tp += 1
            self.iou_sum += float(iou)

            det = known_dets[det_idx]
            g = known_gt[gt_idx]

            self.calibration.append((float(det["confidence"]), 1.0))

            track_id = det.get("track_id")
            key = g["label"]

            if track_id is not None:
                if key in self.gt_track_map and self.gt_track_map[key] != track_id:
                    self.id_switches += 1
                self.gt_track_map[key] = track_id

        for det_idx, det in enumerate(known_dets):
            if det_idx not in matched_det:
                self.fp += 1
                self.calibration.append((float(det["confidence"]), 0.0))

        self.fn += max(0, len(known_gt) - len(matched_gt))

        if len(self.calibration) > 10000:
            self.calibration = self.calibration[-10000:]

        unknown_gt = [g for g in gt if not g.get("known", False)]
        unknown_dets = [d for d in detections if not d.get("known", False)]

        unknown_pairs = []

        for det_idx, det in enumerate(unknown_dets):
            for gt_idx, g in enumerate(unknown_gt):
                iou = bbox_iou(det["bbox"], g["bbox"])
                if iou >= 0.25:
                    unknown_pairs.append((iou, det_idx, gt_idx))

        unknown_pairs.sort(key=lambda x: x[0], reverse=True)

        matched_unknown_det = set()
        matched_unknown_gt = set()

        for iou, det_idx, gt_idx in unknown_pairs:
            if det_idx in matched_unknown_det or gt_idx in matched_unknown_gt:
                continue

            matched_unknown_det.add(det_idx)
            matched_unknown_gt.add(gt_idx)

            det = unknown_dets[det_idx]
            g = unknown_gt[gt_idx]

            cluster_id = det.get("cluster_id")
            if cluster_id is not None:
                self.unknown_votes[g["label"]][cluster_id] += 1
                self.unknown_matched += 1

    def metrics(self):
        precision = self.tp / float(self.tp + self.fp) if self.tp + self.fp > 0 else 0.0
        recall = self.tp / float(self.tp + self.fn) if self.tp + self.fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        mean_iou = self.iou_sum / float(self.tp) if self.tp > 0 else 0.0

        ece = 0.0

        if self.calibration:
            bins = 10
            total = len(self.calibration)
            confs = np.array([c for c, _ in self.calibration], dtype=np.float32)
            truths = np.array([y for _, y in self.calibration], dtype=np.float32)

            for b in range(bins):
                low = b / bins
                high = (b + 1) / bins

                if b == bins - 1:
                    mask = (confs >= low) & (confs <= high)
                else:
                    mask = (confs >= low) & (confs < high)

                if mask.sum() == 0:
                    continue

                bin_conf = float(confs[mask].mean())
                bin_acc = float(truths[mask].mean())
                weight = float(mask.sum()) / total
                ece += abs(bin_conf - bin_acc) * weight

        total_votes = 0
        correct_votes = 0

        for true_label, cluster_counts in self.unknown_votes.items():
            if cluster_counts:
                total_votes += sum(cluster_counts.values())
                correct_votes += max(cluster_counts.values())

        unknown_cluster_accuracy = correct_votes / float(total_votes) if total_votes > 0 else 0.0

        cluster_totals = defaultdict(int)
        cluster_true_votes = defaultdict(lambda: defaultdict(int))

        for true_label, cluster_counts in self.unknown_votes.items():
            for cluster_id, count in cluster_counts.items():
                cluster_totals[cluster_id] += count
                cluster_true_votes[cluster_id][true_label] += count

        purity_correct = 0

        for cluster_id, total in cluster_totals.items():
            if cluster_true_votes[cluster_id]:
                purity_correct += max(cluster_true_votes[cluster_id].values())

        unknown_cluster_purity = purity_correct / float(total_votes) if total_votes > 0 else 0.0

        return {
            "frames": self.frames,
            "known_tp": self.tp,
            "known_fp": self.fp,
            "known_fn": self.fn,
            "known_precision": round(precision, 4),
            "known_recall": round(recall, 4),
            "known_f1": round(f1, 4),
            "known_mean_iou": round(mean_iou, 4),
            "expected_calibration_error": round(ece, 4),
            "id_switches": self.id_switches,
            "unknown_matched_detections": self.unknown_matched,
            "unknown_cluster_accuracy": round(unknown_cluster_accuracy, 4),
            "unknown_cluster_purity": round(unknown_cluster_purity, 4),
        }


def annotate_frame(frame, detections, config, metrics, frame_id, timestamp):
    overlay = frame.copy()
    height, width = overlay.shape[:2]

    for det in detections:
        x, y, w, h = [int(v) for v in det["bbox"]]
        x = clamp(x, 0, width - 1)
        y = clamp(y, 0, height - 1)
        w = clamp(w, 1, width - x)
        h = clamp(h, 1, height - y)

        if config.get("privacy_redaction", False):
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
            draw_label(overlay, "REDACTED", (x, max(12, y - 4)), (200, 200, 200))
            continue

        low_conf = bool(det.get("low_conf", False))

        if det.get("known", False):
            color = (0, 200, 0)
        elif low_conf:
            color = (0, 165, 255)
        else:
            color = (255, 80, 255)

        thickness = 1 if low_conf else 2

        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)

        label = f"{det.get('label', 'unknown')} {float(det.get('confidence', 0.0)):.2f}"

        if det.get("track_id") is not None:
            label += f" T{det['track_id']}"

        if not det.get("known", False) and det.get("cluster_id") is not None:
            label += f" C{det['cluster_id']}"

        draw_label(overlay, label, (x, max(12, y - 4)), color)

    hud = (
        f"Frame {frame_id} | "
        f"Known {metrics.get('known_count', 0)} | "
        f"Novel clusters {metrics.get('unknown_clusters', 0)} | "
        f"FPS {metrics.get('fps', 0.0):.1f} | "
        f"Lat {metrics.get('latency_ms', 0.0):.0f} ms | "
        f"Low-conf queue {metrics.get('review', 0)}"
    )

    cv2.rectangle(overlay, (0, 0), (width, 24), (20, 20, 20), -1)
    cv2.putText(
        overlay,
        hud,
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return overlay


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

class PerceptionPipeline:
    def __init__(self, config, source="demo", upload_path=None, seed=42):
        self.config = config
        self.source_kind = source
        self.frame_idx = 0
        self.start_time = time.time()
        self.last_process_time = None

        self.fps_meter = deque(maxlen=60)
        self.latency_ms = deque(maxlen=60)

        self.events = deque(maxlen=200)
        self.detections_log = deque(maxlen=5000)
        self.review_log = deque(maxlen=500)
        self.history = deque(maxlen=int(config.get("max_history", 500)))

        self.status = {
            "errors": 0,
            "warnings": 0,
            "skipped": 0,
        }

        self.discovery = IncrementalDiscovery(
            similarity_threshold=float(config.get("novelty_similarity", 0.74)),
            max_clusters=24,
        )

        self.tracker = SimpleTracker()
        self.evaluator = EvaluationAccumulator()

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=28,
            detectShadows=True,
        )

        try:
            self.source = self._make_source(source, upload_path, seed)
        except Exception as exc:
            self.status["errors"] += 1
            self.add_event("error", f"Source initialization failed: {exc}. Falling back to demo.", "error")
            self.source = SyntheticDemoSource(
                width=int(config.get("width", 640)),
                height=int(config.get("height", 360)),
                fps=float(config.get("fps", 12)),
                seed=seed,
            )

        self.add_event("info", "Pipeline initialized.")

    def _make_source(self, source, upload_path, seed):
        width = int(self.config.get("width", 640))
        height = int(self.config.get("height", 360))
        fps = float(self.config.get("fps", 12))

        if source == "demo":
            return SyntheticDemoSource(width=width, height=height, fps=fps, seed=seed)

        if source == "video":
            if not upload_path or not os.path.exists(upload_path):
                raise IOError("Uploaded video file is missing.")
            return FileVideoSource(upload_path, width=width, height=height, fps=fps)

        if source == "webcam":
            return WebcamSource(width=width, height=height, index=0)

        return SyntheticDemoSource(width=width, height=height, fps=fps, seed=seed)

    def add_event(self, event_type, message, severity="info"):
        self.events.append(
            {
                "utc": utc_now_iso(),
                "frame_id": self.frame_idx,
                "type": event_type,
                "message": message,
                "severity": severity,
            }
        )

    def next_frame(self):
        try:
            frame, gt, timestamp = self.source.read()

            if frame is None:
                return None, None, None

            if not hasattr(frame, "size") or frame.size == 0:
                self.status["skipped"] += 1
                self.add_event("warning", "Empty/corrupt frame skipped.", "warning")
                return None, None, None

            return frame, gt, timestamp

        except Exception as exc:
            self.status["errors"] += 1
            self.add_event("error", f"Frame acquisition failed: {exc}", "error")
            return None, None, None

    def _log_detection(self, det, timestamp):
        velocity = det.get("velocity_px_s", (0.0, 0.0))

        if isinstance(velocity, (tuple, list)) and len(velocity) == 2:
            vx, vy = float(velocity[0]), float(velocity[1])
        else:
            vx, vy = 0.0, 0.0

        measurements = det.get("measurements", {})

        self.detections_log.append(
            {
                "utc": utc_now_iso(),
                "frame_id": self.frame_idx,
                "timestamp": float(timestamp or 0.0),
                "track_id": det.get("track_id"),
                "label": det.get("label", "unknown"),
                "known": bool(det.get("known", False)),
                "confidence": round(float(det.get("confidence", 0.0)), 4),
                "x": int(det["bbox"][0]),
                "y": int(det["bbox"][1]),
                "w": int(det["bbox"][2]),
                "h": int(det["bbox"][3]),
                "cluster_id": det.get("cluster_id"),
                "low_conf": bool(det.get("low_conf", False)),
                "area_px": float(measurements.get("area_px", 0.0)),
                "aspect": float(measurements.get("aspect", 0.0)),
                "solidity": float(measurements.get("solidity", 0.0)),
                "velocity_px_s": round(math.hypot(vx, vy), 2),
                "evidence": det.get("evidence", ""),
            }
        )

    def current_metrics(self, known_count=None, avg_conf=None):
        fps = float(np.mean(self.fps_meter)) if self.fps_meter else float(self.config.get("fps", 12))
        latency = float(np.mean(self.latency_ms)) if self.latency_ms else 0.0

        if known_count is None:
            known_count = 0
            if self.history:
                known_count = int(self.history[-1].get("known", 0))

        if avg_conf is None:
            avg_conf = 0.0
            if self.history:
                avg_conf = float(self.history[-1].get("avg_conf", 0.0))

        return {
            "fps": fps,
            "latency_ms": latency,
            "known_count": known_count,
            "unknown_clusters": len(self.discovery.clusters),
            "avg_conf": avg_conf,
            "review": len(self.review_log),
            "errors": self.status["errors"],
            "warnings": self.status["warnings"],
            "skipped": self.status["skipped"],
        }

    def process_frame(self, frame, gt, timestamp):
        start = time.perf_counter()

        if frame is None:
            return None

        target_width = int(self.config.get("width", 640))
        h, w = frame.shape[:2]

        if w > target_width * 1.05:
            new_h = int(h * target_width / w)
            frame = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)

        h, w = frame.shape[:2]

        if min(h, w) < 64:
            self.status["skipped"] += 1
            self.add_event("warning", "Insufficient resolution. Frame skipped.", "warning")
            return None

        try:
            known_dets, known_mask = detect_known(frame, self.config)
            unknown_dets = detect_unknown(frame, known_mask, self.bg_subtractor, self.config)
        except Exception as exc:
            self.status["errors"] += 1
            self.add_event("error", f"Model/inference failure: {exc}", "error")
            known_dets = []
            unknown_dets = []

        detections = known_dets + unknown_dets

        # Open-world discovery for unknown detections
        for det in detections:
            if det.get("known", False):
                continue

            embedding = compute_embedding(frame, det["bbox"])
            cluster_id, novelty, is_new = self.discovery.assign(embedding)

            det["cluster_id"] = cluster_id
            det["novelty"] = float(novelty)
            det["label"] = self.discovery.clusters[cluster_id]["label"]

            if is_new:
                self.add_event(
                    "discovery",
                    f"New unseen category candidate: {det['label']} (novelty={novelty:.2f})",
                    "info",
                )

        dt = 1.0 / max(1.0, float(self.config.get("fps", 12)))
        if self.last_process_time is not None:
            dt = max(0.01, time.perf_counter() - self.last_process_time)

        detections = self.tracker.update(detections, w, h, dt)

        conf_threshold = float(self.config.get("confidence_threshold", 0.45))
        known_count = 0

        for det in detections:
            if det.get("known", False):
                known_count += 1
            else:
                track = self.tracker.tracks.get(det.get("track_id"))

                if track and track.get("cluster_votes"):
                    majority_cluster = max(
                        track["cluster_votes"].items(),
                        key=lambda kv: (kv[1], -kv[0]),
                    )[0]

                    if track["hits"] >= 2 and majority_cluster in self.discovery.clusters:
                        det["cluster_id"] = majority_cluster
                        det["label"] = self.discovery.clusters[majority_cluster]["label"]

            det["low_conf"] = bool(float(det.get("confidence", 0.0)) < conf_threshold)

            if det["low_conf"]:
                self.review_log.append(
                    {
                        "utc": utc_now_iso(),
                        "frame_id": self.frame_idx,
                        "label": det.get("label", "unknown"),
                        "known": bool(det.get("known", False)),
                        "confidence": round(float(det.get("confidence", 0.0)), 4),
                        "bbox": [int(v) for v in det["bbox"]],
                        "track_id": det.get("track_id"),
                        "cluster_id": det.get("cluster_id"),
                        "reason": "low_confidence",
                    }
                )

            x, y, bw, bh = [int(v) for v in det["bbox"]]
            det["evidence"] = f"frame:{self.frame_idx};bbox:{x},{y},{bw},{bh}"

        eval_dets = detections

        detections = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
        max_detections = int(self.config.get("max_detections", 20))
        display_dets = detections[:max_detections]

        for det in display_dets:
            self._log_detection(det, timestamp)

        if gt:
            try:
                self.evaluator.update(eval_dets, gt)
            except Exception as exc:
                self.status["warnings"] += 1
                self.add_event("warning", f"Evaluation warning: {exc}", "warning")

        avg_conf = float(np.mean([d.get("confidence", 0.0) for d in eval_dets])) if eval_dets else 0.0

        metrics = self.current_metrics(known_count, avg_conf)
        annotated = annotate_frame(frame, display_dets, self.config, metrics, self.frame_idx, timestamp)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        now = time.perf_counter()

        inst_fps = float(self.config.get("fps", 12))
        if self.last_process_time is not None:
            inst_fps = 1.0 / max(1e-6, now - self.last_process_time)

        self.last_process_time = now
        self.fps_meter.append(inst_fps)
        self.latency_ms.append(elapsed_ms)

        metrics = self.current_metrics(known_count, avg_conf)

        self.history.append(
            {
                "frame": self.frame_idx,
                "time": round(float(timestamp or 0.0), 3),
                "fps": round(metrics["fps"], 2),
                "latency_ms": round(metrics["latency_ms"], 1),
                "known": known_count,
                "clusters": len(self.discovery.clusters),
                "avg_conf": round(avg_conf, 3),
            }
        )

        self.frame_idx += 1

        return annotated


# ------------------------------------------------------------
# Demo video export
# ------------------------------------------------------------

def generate_demo_video(seconds, fps, width, config):
    seconds = int(clamp(seconds, 1, 60))
    fps = int(clamp(fps, 4, 24))
    width = int(clamp(width, 320, 640))
    height = int(width * 9 / 16)

    export_cfg = config.copy()
    export_cfg.update(
        {
            "width": width,
            "height": height,
            "fps": fps,
            "batch_frames": 1,
        }
    )

    pipeline = PerceptionPipeline(export_cfg, source="demo", seed=2026)

    n_frames = int(seconds * fps)
    tmp_path = os.path.join(tempfile.gettempdir(), f"openworld_demo_{int(time.time())}.mp4")

    writer = None
    use_imageio = True

    try:
        import imageio

        writer = imageio.get_writer(
            tmp_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )

        def write_frame(img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb)

    except Exception:
        use_imageio = False
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

        def write_frame(img):
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height))
            writer.write(img)

    progress = st.progress(0.0, text="Rendering demo video...")

    for i in range(n_frames):
        frame, gt, ts = pipeline.next_frame()

        if frame is None:
            break

        annotated = pipeline.process_frame(frame, gt, ts)

        if annotated is None:
            continue

        if annotated.shape[1] != width or annotated.shape[0] != height:
            annotated = cv2.resize(annotated, (width, height))

        write_frame(annotated)
        progress.progress((i + 1) / float(n_frames), text=f"Rendering frame {i + 1}/{n_frames}")

    if use_imageio:
        writer.close()
    else:
        writer.release()

    with open(tmp_path, "rb") as f:
        video_bytes = f.read()

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    progress.empty()

    return video_bytes


def run_benchmark(config, frames=180):
    bench_cfg = config.copy()
    bench_cfg["width"] = min(int(config.get("width", 640)), 480)
    bench_cfg["height"] = int(bench_cfg["width"] * 9 / 16)

    pipeline = PerceptionPipeline(bench_cfg, source="demo", seed=7)

    progress = st.progress(0.0, text="Running evaluation benchmark...")

    for i in range(int(frames)):
        frame, gt, ts = pipeline.next_frame()

        if frame is None:
            break

        pipeline.process_frame(frame, gt, ts)
        progress.progress((i + 1) / float(frames), text=f"Benchmark frame {i + 1}/{frames}")

    progress.empty()

    return pipeline.evaluator.metrics(), pipeline.current_metrics()


# ------------------------------------------------------------
# Streamlit UI state and controls
# ------------------------------------------------------------

if "running" not in st.session_state:
    st.session_state.running = True

if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

if "demo_video" not in st.session_state:
    st.session_state.demo_video = None

with st.sidebar:
    st.header("⚙️ Configuration")

    source_option = st.selectbox(
        "Input source",
        ["Deterministic demo scene", "Uploaded video", "Webcam"],
        index=0,
    )

    uploaded_file = None
    upload_path = None

    if source_option == "Uploaded video":
        uploaded_file = st.file_uploader(
            "Upload video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
        )

        if uploaded_file is not None:
            if st.session_state.get("upload_name") != uploaded_file.name:
                suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded_file.getvalue())
                tmp.close()

                st.session_state.upload_path = tmp.name
                st.session_state.upload_name = uploaded_file.name

            upload_path = st.session_state.get("upload_path")
        else:
            st.info("No video uploaded yet. The app will fall back to demo mode.")

    st.subheader("Resolution and performance")

    width = st.select_slider(
        "Resolution width",
        options=[320, 480, 640, 800, 1280],
        value=640,
    )

    height = int(width * 9 / 16)

    fps = st.slider("Demo FPS", 4, 24, 12)

    refresh_choice = st.selectbox(
        "Live refresh rate",
        ["Fast", "Normal", "Slow"],
        index=1,
    )

    refresh_sec = {
        "Fast": 0.25,
        "Normal": 0.45,
        "Slow": 0.80,
    }[refresh_choice]

    batch_frames = st.selectbox("Frames per live tick", [1, 2, 3], index=1)

    st.subheader("Perception thresholds")

    conf_threshold = st.slider(
        "Known-object confidence threshold",
        0.05,
        0.95,
        0.45,
        0.05,
    )

    novelty_similarity = st.slider(
        "Novelty similarity threshold",
        0.50,
        0.95,
        0.74,
        0.01,
        help="Higher values make new unseen clusters easier to create.",
    )

    max_detections = st.slider("Max displayed detections", 5, 50, 20)

    privacy_redaction = st.checkbox(
        "Privacy redaction mode",
        value=False,
        help="Redacts detected regions. No identity recognition is used.",
    )

    auto_start = st.checkbox("Auto-start live simulation", value=True)

    st.divider()

    if st.button("⏯ Start / Pause live run", use_container_width=True):
        st.session_state.running = not st.session_state.running
        st.rerun()

    if st.button("♻️ Reset pipeline", use_container_width=True):
        st.session_state.reset_pipeline = True
        st.session_state.last_frame = None
        st.session_state.demo_video = None
        st.rerun()

    st.divider()
    st.subheader("🎬 Demo video export")

    video_seconds = st.slider("Video length (seconds)", 5, 60, 30)
    video_fps = st.select_slider("Export FPS", options=[6, 8, 10, 12], value=8)

    if st.button("Generate live demo MP4", type="primary", use_container_width=True):
        with st.status("Rendering demo video...", expanded=True) as status:
            try:
                export_width = min(int(width), 480)
                video_bytes = generate_demo_video(
                    seconds=video_seconds,
                    fps=int(video_fps),
                    width=export_width,
                    config=st.session_state.get("config", {}),
                )
                st.session_state.demo_video = video_bytes
                status.update(label="Demo video ready", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Video generation failed", state="error", expanded=True)
                st.error(f"Video generation failed: {exc}")

config = {
    "width": int(width),
    "height": int(height),
    "fps": int(fps),
    "confidence_threshold": float(conf_threshold),
    "novelty_similarity": float(novelty_similarity),
    "max_detections": int(max_detections),
    "privacy_redaction": bool(privacy_redaction),
    "batch_frames": int(batch_frames),
    "min_candidate_conf": 0.12,
    "max_history": 500,
    "auto_start": bool(auto_start),
}

st.session_state.config = config

source_sig = "|".join(
    [
        source_option,
        uploaded_file.name if uploaded_file is not None else "none",
        str(width),
        str(height),
        str(fps),
    ]
)

reset_requested = st.session_state.pop("reset_pipeline", False)

pipeline_source = "demo"
if source_option == "Uploaded video":
    pipeline_source = "video"
elif source_option == "Webcam":
    pipeline_source = "webcam"

if (
    "pipeline" not in st.session_state
    or reset_requested
    or st.session_state.get("source_sig") != source_sig
):
    st.session_state.pipeline = PerceptionPipeline(
        config=config,
        source=pipeline_source,
        upload_path=upload_path,
        seed=42,
    )
    st.session_state.source_sig = source_sig
    st.session_state.running = bool(auto_start)
    st.session_state.last_frame = None
else:
    st.session_state.pipeline.config = config


pipeline = st.session_state.pipeline


# ------------------------------------------------------------
# Live rendering
# ------------------------------------------------------------

def render_live():
    pipe = st.session_state.pipeline

    if st.session_state.running:
        batch = int(st.session_state.config.get("batch_frames", 1))

        for _ in range(batch):
            try:
                frame, gt, ts = pipe.next_frame()

                if frame is None:
                    st.session_state.running = False
                    break

                annotated = pipe.process_frame(frame, gt, ts)

                if annotated is not None:
                    st.session_state.last_frame = annotated

            except Exception as exc:
                pipe.status["errors"] += 1
                pipe.add_event("error", f"Live loop error: {exc}", "error")
                st.session_state.running = False
                break

    if st.session_state.get("demo_video") is not None:
        with st.expander("🎬 Generated demo video", expanded=False):
            st.video(st.session_state.demo_video)
            st.download_button(
                "Download MP4 demo video",
                data=st.session_state.demo_video,
                file_name="openworldperception_demo.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

    if st.session_state.get("last_frame") is not None:
        st.image(
            st.session_state.last_frame,
            caption=f"Live perception frame {pipe.frame_idx}",
            use_container_width=True,
        )
    else:
        st.info("Starting live demo...")

    metrics = pipe.current_metrics()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("FPS", f"{metrics['fps']:.1f}")
    c2.metric("Latency (ms)", f"{metrics['latency_ms']:.0f}")
    c3.metric("Known detections", metrics["known_count"])
    c4.metric("Novel clusters", metrics["unknown_clusters"])
    c5.metric("Low-conf review", metrics["review"])

    if pipe.status["errors"] > 0:
        st.warning(f"Pipeline errors detected: {pipe.status['errors']}. See events tab for details.")

    hist = list(pipe.history)[-120:]

    if hist:
        df = pd.DataFrame(hist)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Performance**")
            chart_df = df.set_index("frame")[["fps", "latency_ms"]]
            st.line_chart(chart_df, use_container_width=True)

        with col_b:
            st.markdown("**Perception state**")
            chart_df = df.set_index("frame")[["known", "clusters", "avg_conf"]]
            st.line_chart(chart_df, use_container_width=True)

    st.markdown("**Recent events**")

    events = list(pipe.events)[-8:][::-1]

    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
    else:
        st.caption("No events yet.")


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------

tab_live, tab_discovery, tab_eval, tab_export, tab_about = st.tabs(
    [
        "📺 Live Dashboard",
        "🧠 Novel Discovery",
        "📊 Evaluation",
        "💾 Evidence / Export",
        "ℹ️ About",
    ]
)

with tab_live:
    if hasattr(st, "fragment"):
        live_fragment = st.fragment(run_every=refresh_sec)(render_live)
        live_fragment()
    else:
        render_live()
        if st.button("Manual refresh"):
            st.rerun()

with tab_discovery:
    st.subheader("Discovered unseen categories")

    clusters = list(pipeline.discovery.clusters.values())

    if clusters:
        cluster_rows = []

        for cluster in clusters:
            cluster_rows.append(
                {
                    "cluster_id": cluster["id"],
                    "label": cluster["label"],
                    "observations": cluster["count"],
                    "created_utc": cluster["created_utc"],
                }
            )

        st.dataframe(pd.DataFrame(cluster_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No unseen categories discovered yet.")

    st.subheader("Low-confidence review queue")

    review = list(pipeline.review_log)[-50:][::-1]

    if review:
        st.dataframe(pd.DataFrame(review), use_container_width=True, hide_index=True)
    else:
        st.info("No low-confidence items queued.")

with tab_eval:
    st.subheader("Live evaluation on deterministic demo ground truth")

    live_metrics = pipeline.evaluator.metrics()

    if live_metrics.get("frames", 0) > 0:
        st.json(live_metrics)
    else:
        st.info("No evaluation accumulated yet. Let the live demo run or run the benchmark below.")

    st.divider()

    benchmark_frames = st.slider("Benchmark frames", 60, 600, 180, 60)

    if st.button("Run deterministic benchmark", type="primary"):
        bench_metrics, runtime_metrics = run_benchmark(st.session_state.config, frames=benchmark_frames)

        st.success("Benchmark complete.")
        st.json(
            {
                "evaluation": bench_metrics,
                "runtime": runtime_metrics,
            }
        )

with tab_export:
    st.subheader("Structured evidence and results")

    detections = list(pipeline.detections_log)[-1000:]

    if detections:
        det_df = pd.DataFrame(detections)
        st.dataframe(det_df.tail(200), use_container_width=True, hide_index=True)

        csv_bytes = det_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download detections CSV",
            data=csv_bytes,
            file_name="openworldperception_detections.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No detections logged yet.")

    events = list(pipeline.events)

    if events:
        events_json = json.dumps(events, indent=2, default=str)

        st.download_button(
            "Download events JSON",
            data=events_json.encode("utf-8"),
            file_name="openworldperception_events.json",
            mime="application/json",
            use_container_width=True,
        )

    report = {
        "generated_utc": utc_now_iso(),
        "config": {
            k: v for k, v in st.session_state.config.items() if k not in ["upload_path"]
        },
        "status": pipeline.status,
        "metrics": pipeline.evaluator.metrics(),
        "novel_clusters": [
            {
                "id": c["id"],
                "label": c["label"],
                "observations": c["count"],
                "created_utc": c["created_utc"],
            }
            for c in pipeline.discovery.clusters.values()
        ],
    }

    st.download_button(
        "Download full report JSON",
        data=json.dumps(report, indent=2, default=str).encode("utf-8"),
        file_name="openworldperception_report.json",
        mime="application/json",
        use_container_width=True,
    )

with tab_about:
    st.markdown(
        """
        ## OpenWorldPerception AI

        Production-style prototype for open-world recognition.

        ### What it does
        - Detects known objects in a deterministic demo scene.
        - Discovers previously unseen object categories.
        - Clusters unknown observations incrementally.
        - Applies temporal tracking and track-level cluster smoothing.
        - Routes low-confidence detections to a review queue.
        - Provides live FPS, latency, confidence, and discovery metrics.
        - Exports structured evidence and evaluation reports.

        ### Distinctive advanced capability
        The distinctive capability is **incremental open-world discovery**:
        unknown regions are embedded using appearance features and assigned
        to discovered categories using nearest-centroid novelty clustering.

        ### Privacy
        - No face recognition.
        - No identity inference.
        - Optional redaction mode.
        - Local processing only.

        ### Notes
        - The demo mode is deterministic and synthetic.
        - Webcam/uploaded-video modes are optional.
        - No paid API key is required.
        """
    )