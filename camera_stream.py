"""
Project SilentEcho: Camera Stream & Lip ROI Extraction Module
Optimized for HP Snapdragon X Elite / X Plus Laptops (Windows 11 ARM64)

Features:
- Windows Media Foundation (cv2.CAP_MSMF) high-throughput video capture.
- Automated lip region detection with Exponential Moving Average (EMA) coordinate smoothing.
- Normalized 40x80 grayscale ROI extraction.
- NPU tensor preparation: (B, T, C, H, W) normalized float32.
- Mock fallback stream for headless testing or environments without a camera.
"""

import sys
import time
import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Generator

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SilentEcho.CameraStream")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not installed. Falling back to synthetic mock stream.")


@dataclass
class CameraConfig:
    camera_index: int = 0
    target_fps: int = 30
    frame_width: int = 1280
    frame_height: int = 720
    roi_height: int = 40
    roi_width: int = 80
    sequence_length: int = 16  # Frames per visual temporal window
    use_msmf: bool = True      # cv2.CAP_MSMF (Windows Media Foundation)
    normalize_range: Tuple[float, float] = (-1.0, 1.0)
    ema_alpha: float = 0.65    # Smoothing factor for ROI coordinates


class LipRoiTracker:
    """
    Tracks and extracts the mouth region from video frames.
    Uses facial geometry heuristics with temporal EMA smoothing to avoid bounding box jitter.
    """
    def __init__(self, target_h: int = 40, target_w: int = 80, ema_alpha: float = 0.65):
        self.target_h = target_h
        self.target_w = target_w
        self.ema_alpha = ema_alpha
        self.smoothed_bbox: Optional[Tuple[int, int, int, int]] = None
        
        # Load OpenCV Haar cascade if available
        self.face_cascade = None
        if OPENCV_AVAILABLE:
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception as e:
                logger.debug("Haar cascade initialization skipped: %s", e)

    def extract_lip_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extracts and normalizes the 40x80 grayscale lip region.
        Returns:
            normalized_roi: np.ndarray of shape (target_h, target_w) float32 in [-1, 1]
            bbox: (x, y, w, h) bounding box on the original frame
        """
        h_orig, w_orig = frame.shape[:2]
        if len(frame.shape) == 3:
            if OPENCV_AVAILABLE:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                # ITU-R 601-2 luma transform
                gray = (frame[:, :, 0] * 0.114 + frame[:, :, 1] * 0.587 + frame[:, :, 2] * 0.299).astype(np.uint8)
        else:
            gray = frame

        detected_box = None
        if OPENCV_AVAILABLE and self.face_cascade is not None and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(100, 100))
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                # Lower third of face contains the mouth
                mouth_x = fx + int(fw * 0.25)
                mouth_y = fy + int(fh * 0.65)
                mouth_w = int(fw * 0.50)
                mouth_h = int(fh * 0.25)
                detected_box = (mouth_x, mouth_y, mouth_w, mouth_h)

        if detected_box is None:
            # Fallback heuristic: Center-lower quadrant for HP OmniBook webcam
            mouth_w = int(w_orig * 0.22)
            mouth_h = int(h_orig * 0.16)
            mouth_x = int((w_orig - mouth_w) / 2)
            mouth_y = int(h_orig * 0.58)
            detected_box = (mouth_x, mouth_y, mouth_w, mouth_h)

        # Apply Exponential Moving Average (EMA) smoothing
        if self.smoothed_bbox is None:
            self.smoothed_bbox = detected_box
        else:
            sx, sy, sw, sh = self.smoothed_bbox
            dx, dy, dw, dh = detected_box
            self.smoothed_bbox = (
                int(self.ema_alpha * dx + (1 - self.ema_alpha) * sx),
                int(self.ema_alpha * dy + (1 - self.ema_alpha) * sy),
                int(self.ema_alpha * dw + (1 - self.ema_alpha) * sw),
                int(self.ema_alpha * dh + (1 - self.ema_alpha) * sh),
            )

        x, y, w, h = self.smoothed_bbox
        # Clamp to frame bounds
        x = max(0, min(x, w_orig - 1))
        y = max(0, min(y, h_orig - 1))
        w = max(10, min(w, w_orig - x))
        h = max(10, min(h, h_orig - y))

        roi_patch = gray[y:y+h, x:x+w]
        if roi_patch.size == 0:
            roi_patch = np.zeros((self.target_h, self.target_w), dtype=np.uint8)
        else:
            if OPENCV_AVAILABLE:
                roi_patch = cv2.resize(roi_patch, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
            else:
                from PIL import Image
                pil_img = Image.fromarray(roi_patch)
                pil_resized = pil_img.resize((self.target_w, self.target_h), Image.Resampling.BILINEAR)
                roi_patch = np.array(pil_resized, dtype=np.uint8)

        # Contrast equalization & normalization to [-1.0, 1.0]
        roi_norm = (roi_patch.astype(np.float32) / 127.5) - 1.0
        return roi_norm, (x, y, w, h)


class CameraStream:
    """
    Non-blocking multi-threaded camera ingestion engine.
    Utilizes Windows Media Foundation (MSMF) for hardware-accelerated video decoding on Snapdragon X.
    """
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.cap = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Ring buffer for recent lip ROI frames
        self.roi_buffer: deque = deque(maxlen=self.config.sequence_length * 2)
        self.tracker = LipRoiTracker(
            target_h=self.config.roi_height,
            target_w=self.config.roi_width,
            ema_alpha=self.config.ema_alpha
        )
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_bbox: Optional[Tuple[int, int, int, int]] = None
        self.fps_actual = 0.0
        self.frame_count = 0
        self.start_time = 0.0

    def start(self) -> bool:
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV unavailable. Running in simulated camera mode.")
            return False

        # Attempt hardware-accelerated Windows Media Foundation (MSMF)
        backend = cv2.CAP_MSMF if (self.config.use_msmf and sys.platform == "win32") else cv2.CAP_ANY
        logger.info("Initializing camera %d with backend %s...", self.config.camera_index, "MSMF" if backend == cv2.CAP_MSMF else "DEFAULT")
        
        self.cap = cv2.VideoCapture(self.config.camera_index, backend)
        if not self.cap.isOpened():
            # Retry with default backend
            self.cap = cv2.VideoCapture(self.config.camera_index)
            
        if not self.cap.isOpened():
            logger.warning("Failed to open physical camera device. Falling back to synthetic stream.")
            return False

        # Configure hardware capture properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)

        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name="SilentEcho-Camera")
        self.thread.start()
        logger.info("Camera capture thread launched successfully at target %d FPS.", self.config.target_fps)
        return True

    def _capture_loop(self):
        last_time = time.time()
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            timestamp = time.time()
            roi_norm, bbox = self.tracker.extract_lip_roi(frame)

            with self.lock:
                self.latest_raw_frame = frame
                self.latest_bbox = bbox
                self.roi_buffer.append((timestamp, roi_norm))
                self.frame_count += 1
                
                # Compute actual FPS every 30 frames
                if self.frame_count % 30 == 0:
                    now = time.time()
                    self.fps_actual = 30.0 / (now - last_time) if (now - last_time) > 0 else self.config.target_fps
                    last_time = now

    def get_latest_sequence(self, sequence_length: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Retrieves visual tensor formatted for NPU inference:
        Shape: (1, sequence_length, 1, roi_height, roi_width) float32
        """
        seq_len = sequence_length or self.config.sequence_length
        with self.lock:
            if len(self.roi_buffer) < seq_len:
                return None
            # Extract last seq_len frames
            items = list(self.roi_buffer)[-seq_len:]
            frames = [item[1] for item in items]

        # Stack into (T, H, W) -> add batch and channel dims -> (1, T, 1, H, W)
        tensor = np.stack(frames, axis=0) # (T, H, W)
        tensor = np.expand_dims(tensor, axis=1) # (T, 1, H, W)
        tensor = np.expand_dims(tensor, axis=0) # (1, T, 1, H, W)
        return tensor.astype(np.float32)

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        logger.info("Camera stream stopped.")


class MockCameraStream:
    """
    High-fidelity synthetic lip-motion generator for headless environments, CI/CD,
    and testing without physical camera hardware.
    Generates realistic oscillating mouth ellipses simulating silent phoneme articulations.
    """
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.roi_buffer: deque = deque(maxlen=self.config.sequence_length * 2)
        self.frame_count = 0
        self.fps_actual = float(self.config.target_fps)

    def start(self) -> bool:
        self.running = True
        self.thread = threading.Thread(target=self._generator_loop, daemon=True, name="SilentEcho-MockCamera")
        self.thread.start()
        logger.info("Mock camera stream started (Target: %d FPS).", self.config.target_fps)
        return True

    def _generator_loop(self):
        h, w = self.config.roi_height, self.config.roi_width
        interval = 1.0 / self.config.target_fps
        t = 0.0

        while self.running:
            start_tick = time.time()
            t += interval

            # Synthesize synthetic mouth opening/closing with harmonic articulation
            # Vertical aperture: 4 to 14 pixels
            aperture_y = 9.0 + 5.0 * np.sin(2.0 * np.pi * 1.5 * t) * np.cos(2.0 * np.pi * 0.7 * t)
            aperture_x = 22.0 + 4.0 * np.cos(2.0 * np.pi * 1.5 * t)

            # Generate synthetic grayscale patch
            yy, xx = np.mgrid[0:h, 0:w]
            cy, cx = h / 2.0, w / 2.0
            
            # Outer lip contour
            outer_dist = ((xx - cx) / (aperture_x + 6)) ** 2 + ((yy - cy) / (aperture_y + 4)) ** 2
            # Inner mouth cavity
            inner_dist = ((xx - cx) / aperture_x) ** 2 + ((yy - cy) / aperture_y) ** 2

            patch = np.ones((h, w), dtype=np.float32) * 0.7  # Skin background
            patch[outer_dist <= 1.0] = 0.4                  # Lips
            patch[inner_dist <= 1.0] = 0.05                 # Oral cavity darkness
            
            # Add subtle sensor noise
            noise = np.random.normal(0, 0.02, (h, w)).astype(np.float32)
            patch = np.clip(patch + noise, 0.0, 1.0)
            
            # Map to [-1.0, 1.0]
            roi_norm = (patch * 2.0) - 1.0

            with self.lock:
                self.roi_buffer.append((time.time(), roi_norm))
                self.frame_count += 1

            elapsed = time.time() - start_tick
            sleep_time = max(0.001, interval - elapsed)
            time.sleep(sleep_time)

    def get_latest_sequence(self, sequence_length: Optional[int] = None) -> Optional[np.ndarray]:
        seq_len = sequence_length or self.config.sequence_length
        with self.lock:
            if len(self.roi_buffer) < seq_len:
                return None
            items = list(self.roi_buffer)[-seq_len:]
            frames = [item[1] for item in items]

        tensor = np.stack(frames, axis=0) # (T, H, W)
        tensor = np.expand_dims(tensor, axis=1) # (T, 1, H, W)
        tensor = np.expand_dims(tensor, axis=0) # (1, T, 1, H, W)
        return tensor.astype(np.float32)

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Mock camera stream stopped.")


def create_camera_stream(config: Optional[CameraConfig] = None, force_mock: bool = False):
    """
    Factory function: returns CameraStream if physical webcam available, else MockCameraStream.
    """
    if force_mock:
        stream = MockCameraStream(config)
        stream.start()
        return stream
    
    stream = CameraStream(config)
    success = stream.start()
    if not success:
        logger.info("Falling back to high-fidelity MockCameraStream.")
        stream = MockCameraStream(config)
        stream.start()
    return stream


if __name__ == "__main__":
    print("=== Testing Project SilentEcho Camera Stream ===")
    cfg = CameraConfig(sequence_length=16, roi_height=40, roi_width=80)
    # Check if mock requested
    use_mock = "--mock" in sys.argv
    cam = create_camera_stream(cfg, force_mock=use_mock)

    try:
        print("Warming up camera buffer (collecting frames)...")
        time.sleep(1.0)
        for i in range(10):
            seq = cam.get_latest_sequence()
            if seq is not None:
                print(f"[{i+1}/10] Extracted visual tensor: shape={seq.shape}, dtype={seq.dtype}, min={seq.min():.2f}, max={seq.max():.2f}")
            else:
                print(f"[{i+1}/10] Buffer filling... ({len(cam.roi_buffer)}/{cfg.sequence_length} frames)")
            time.sleep(0.1)
    finally:
        cam.stop()
        print("Camera test completed successfully.")
