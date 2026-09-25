"""
Project SilentEcho: On-Device Multi-Modal Silent Speech & Acoustic Camouflage Engine
Master Runtime Orchestrator & Live Real-Time Dashboard
Optimized for HP Snapdragon X Elite / X Plus Laptops (Windows 11 ARM64)
"""

import os
import sys
import time
import signal
import logging
import argparse
from typing import Optional

import numpy as np

from camera_stream import CameraConfig, create_camera_stream
from audio_stream import AudioConfig, create_audio_stream
from qnn_engine import QNNConfig, SilentEchoQNNEngine
from fusion_decoder import FusionConfig, MultiModalFusionDecoder
from benchmark_metrics import BenchmarkHarness, BenchmarkReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SilentEcho.Main")


class SilentEchoEngine:
    """
    Unified multimodal orchestrator binding camera, HP Poly Studio audio array,
    Qualcomm Hexagon NPU, and real-time fusion decoder into an ultra-low latency pipeline.
    """
    def __init__(
        self,
        use_mock: bool = False,
        provider: str = "auto",
        benchmark_mode: bool = False
    ):
        self.use_mock = use_mock
        self.provider = provider
        self.benchmark_mode = benchmark_mode
        self.running = False

        print("\n" + "=" * 78)
        print("  PROJECT SILENTECHO: ON-DEVICE MULTI-MODAL SILENT SPEECH ENGINE")
        print("  Target: Qualcomm Snapdragon X Series | Qualcomm Hexagon NPU (HTP)")
        print("=" * 78 + "\n")

        # 1. Initialize Visual Pipeline
        logger.info("[1/4] Initializing Windows Media Foundation camera stream...")
        self.cam_config = CameraConfig(sequence_length=16, roi_height=40, roi_width=80)
        self.camera = create_camera_stream(self.cam_config, force_mock=use_mock)

        # 2. Initialize Sub-Vocal Acoustic Pipeline
        logger.info("[2/4] Initializing sub-vocal acoustic stream (16kHz mono)...")
        self.audio_config = AudioConfig(mel_window_frames=64)
        self.audio = create_audio_stream(self.audio_config, force_mock=use_mock)

        # 3. Initialize Qualcomm Hexagon NPU Session
        logger.info("[3/4] Initializing Qualcomm AI Engine Direct (QNN EP) session...")
        self.qnn_config = QNNConfig(
            backend_path="QnnHtp.dll",
            htp_performance_mode="burst",
            htp_graph_finalization_optimization_mode="3",
            enable_htp_fp16_precision="1",
            preferred_provider=provider
        )
        self.npu = SilentEchoQNNEngine(config=self.qnn_config)
        self.hw_info = self.npu.get_hardware_info()

        # 4. Initialize Multi-Modal Fusion Decoder
        logger.info("[4/4] Initializing temporal cross-attention fusion decoder...")
        self.decoder = MultiModalFusionDecoder(on_text_callback=self._on_phrase_decoded)

        # Telemetry & Benchmarking
        self.harness = BenchmarkHarness(target_hardware=self.hw_info["target_silicon"])
        self.transcript_history = []
        self.iteration_count = 0

    def _on_phrase_decoded(self, phrase: str):
        if phrase:
            timestamp_str = time.strftime("%H:%M:%S")
            self.transcript_history.append((timestamp_str, phrase))
            # Keep last 6 phrases for UI
            if len(self.transcript_history) > 6:
                self.transcript_history.pop(0)

    def run(self, duration_sec: Optional[float] = None):
        """
        Executes continuous real-time multi-modal loop.
        """
        self.running = True
        logger.info("Project SilentEcho running. Press Ctrl+C to terminate.\n")

        # Allow hardware sensor buffers to prime
        time.sleep(0.8)

        # Default fallback tensors if sensors are priming
        dummy_v = np.random.uniform(-1.0, 1.0, (1, 16, 1, 40, 80)).astype(np.float32)
        dummy_a = np.random.uniform(-2.0, 2.0, (1, 80, 64)).astype(np.float32)

        start_time = time.time()
        last_ui_update = 0.0

        try:
            while self.running:
                loop_start = time.perf_counter()

                # Stage 1: Visual Ingestion & Lip ROI Extraction
                t0 = time.perf_counter()
                v_tensor = self.camera.get_latest_sequence()
                if v_tensor is None:
                    v_tensor = dummy_v
                t1 = time.perf_counter()
                vis_ms = (t1 - t0) * 1000.0

                # Stage 2: Sub-Vocal Acoustic Ingestion & Log-Mel Spectrogram
                t2 = time.perf_counter()
                a_tensor = self.audio.get_latest_mel_tensor()
                if a_tensor is None:
                    a_tensor = dummy_a
                t3 = time.perf_counter()
                aud_ms = (t3 - t2) * 1000.0

                # Stage 3: Qualcomm Hexagon NPU Inference
                logits, npu_ms = self.npu.infer(v_tensor, a_tensor)

                # Stage 4: Temporal Alignment & CTC Token Decoding
                t4 = time.perf_counter()
                text, conf = self.decoder.decode_logits(logits)
                t5 = time.perf_counter()
                dec_ms = (t5 - t4) * 1000.0

                total_e2e_ms = vis_ms + aud_ms + npu_ms + dec_ms
                self.harness.record_iteration(vis_ms, aud_ms, npu_ms, dec_ms)
                self.iteration_count += 1

                # Live Terminal Dashboard (refresh every ~200ms)
                now = time.time()
                if now - last_ui_update > 0.20:
                    self._render_dashboard(vis_ms, aud_ms, npu_ms, dec_ms, total_e2e_ms, text, conf)
                    last_ui_update = now

                # Duration check
                if duration_sec and (now - start_time) >= duration_sec:
                    break

                # Frame throttle (~30-50 FPS target pace)
                elapsed_loop = time.perf_counter() - loop_start
                sleep_sec = max(0.001, 0.025 - elapsed_loop)
                time.sleep(sleep_sec)

        except KeyboardInterrupt:
            print("\nShutting down SilentEcho Engine gracefully...")
        finally:
            self.stop()

    def _render_dashboard(
        self,
        vis_ms: float,
        aud_ms: float,
        npu_ms: float,
        dec_ms: float,
        e2e_ms: float,
        latest_text: str,
        conf: float
    ):
        """
        Renders an ANSI live status dashboard in the terminal.
        """
        fps = 1000.0 / e2e_ms if e2e_ms > 0 else 0.0
        target_check = "[PASS < 20ms]" if e2e_ms <= 20.0 else "[WARN > 20ms]"
        
        status_lines = [
            "\r" + "-" * 78,
            f"SILENTECHO ACTIVE | Silicon: {self.hw_info['target_silicon']} | EP: {self.hw_info['active_provider']}",
            f"LATENCY: {e2e_ms:5.2f} ms ({fps:4.1f} FPS) {target_check} | NPU: {npu_ms:4.2f}ms | Vision: {vis_ms:4.2f}ms | Audio: {aud_ms:4.2f}ms",
            f"RECENT RECONSTRUCTION (Conf: {conf*100:4.1f}%): '{latest_text if latest_text else '...'}'",
            "-" * 78
        ]
        # In non-interactive or batch terminals, print single compact line
        print(f"[{time.strftime('%H:%M:%S')}] Latency: {e2e_ms:5.2f}ms | NPU: {npu_ms:5.2f}ms | FPS: {fps:4.1f} | Text: '{latest_text or '...'}'")

    def stop(self):
        self.running = False
        self.camera.stop()
        self.audio.stop()
        logger.info("Engine stopped. Processing final performance report...")

        if self.iteration_count > 0:
            report = self.harness.generate_report(active_provider=self.hw_info["active_provider"])
            self.harness.print_summary(report)


def main():
    parser = argparse.ArgumentParser(description="Project SilentEcho: On-Device Multi-Modal Silent Speech Engine")
    parser.add_argument("--mock", action="store_true", help="Force synthetic camera and audio generators")
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "qnn", "dml", "cpu"], help="Execution provider priority")
    parser.add_argument("--duration", type=float, default=None, help="Execution duration in seconds (optional)")
    parser.add_argument("--benchmark", action="store_true", help="Run dedicated benchmark suite and exit")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--json", type=str, default=None, help="Export benchmark metrics to JSON file")
    args = parser.parse_args()

    if args.benchmark:
        from benchmark_metrics import run_standalone_benchmark
        run_standalone_benchmark(iterations=args.iterations, save_json=args.json)
        return

    engine = SilentEchoEngine(
        use_mock=args.mock,
        provider=args.provider,
        benchmark_mode=False
    )
    engine.run(duration_sec=args.duration)


if __name__ == "__main__":
    main()
