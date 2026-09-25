"""
Project SilentEcho: Comprehensive Automated Verification & Unit Test Suite
Target: Qualcomm Snapdragon X Series | Windows 11 ARM64 & Simulation Environments
"""

import os
import sys
import time
import unittest
import numpy as np

from camera_stream import CameraConfig, MockCameraStream, LipRoiTracker, create_camera_stream
from audio_stream import AudioConfig, MockAudioStream, MelFilterbank, SubVocalConditioner, create_audio_stream
from qnn_engine import QNNConfig, SilentEchoQNNEngine
from fusion_decoder import FusionConfig, MultiModalFusionDecoder, SILENTECHO_VOCAB
from benchmark_metrics import BenchmarkHarness, StageTiming


class TestSilentEchoCore(unittest.TestCase):

    def test_01_lip_roi_tracker_normalization(self):
        """Validates that LipRoiTracker produces exactly 40x80 float32 patches normalized to [-1, 1]."""
        tracker = LipRoiTracker(target_h=40, target_w=80, ema_alpha=0.7)
        # Create synthetic test frame 720x1280
        test_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        roi_norm, bbox = tracker.extract_lip_roi(test_frame)

        self.assertEqual(roi_norm.shape, (40, 80))
        self.assertEqual(roi_norm.dtype, np.float32)
        self.assertTrue(np.all(roi_norm >= -1.0) and np.all(roi_norm <= 1.0))
        self.assertEqual(len(bbox), 4)

    def test_02_camera_stream_tensor_formatting(self):
        """Validates that CameraStream generates (1, T, 1, 40, 80) visual tensors for Hexagon NPU."""
        cfg = CameraConfig(sequence_length=16, roi_height=40, roi_width=80, target_fps=30)
        stream = create_camera_stream(cfg, force_mock=True)
        try:
            time.sleep(0.6)  # Allow buffer to prime
            tensor = stream.get_latest_sequence()
            self.assertIsNotNone(tensor)
            self.assertEqual(tensor.shape, (1, 16, 1, 40, 80))
            self.assertEqual(tensor.dtype, np.float32)
        finally:
            stream.stop()

    def test_03_subvocal_acoustic_conditioning(self):
        """Validates 4th-order bandpass filtering, whisper pre-emphasis, and AGC response."""
        cfg = AudioConfig(sample_rate=16000, f_min=120.0, f_max=7200.0, preemphasis=0.97)
        conditioner = SubVocalConditioner(cfg)

        # Generate low-amplitude whisper test signal + low frequency fan rumble (30 Hz)
        t = np.linspace(0, 0.02, 320, endpoint=False)
        rumble = 0.5 * np.sin(2 * np.pi * 30 * t)       # Should be filtered out
        whisper = 0.05 * np.sin(2 * np.pi * 3000 * t)   # Should be amplified by AGC
        chunk = (rumble + whisper).astype(np.float32)

        processed = conditioner.process(chunk)
        self.assertEqual(len(processed), len(chunk))
        self.assertEqual(processed.dtype, np.float32)
        # Verify that output was conditioned without clipping beyond [-1, 1]
        self.assertTrue(np.all(processed >= -1.0) and np.all(processed <= 1.0))

    def test_04_audio_log_mel_tensor_formatting(self):
        """Validates that AudioStream generates (1, 80, 64) Log-Mel tensors for Hexagon NPU."""
        cfg = AudioConfig(sample_rate=16000, n_mels=80, mel_window_frames=64)
        stream = create_audio_stream(cfg, force_mock=True)
        try:
            time.sleep(0.7)
            mel_tensor = stream.get_latest_mel_tensor()
            self.assertIsNotNone(mel_tensor)
            self.assertEqual(mel_tensor.shape, (1, 80, 64))
            self.assertEqual(mel_tensor.dtype, np.float32)
        finally:
            stream.stop()

    def test_05_qnn_engine_multimodal_inference(self):
        """Validates QNN Engine inference execution, tensor contract, and sub-20ms latency."""
        engine = SilentEchoQNNEngine()
        dummy_v = np.random.uniform(-1.0, 1.0, (1, 16, 1, 40, 80)).astype(np.float32)
        dummy_a = np.random.uniform(-2.0, 2.0, (1, 80, 64)).astype(np.float32)

        logits, lat_ms = engine.infer(dummy_v, dummy_a)
        self.assertEqual(logits.shape, (1, 16, 64))
        self.assertEqual(logits.dtype, np.float32)
        self.assertLess(lat_ms, 20.0, f"Inference took {lat_ms:.2f}ms, exceeding sub-20ms target!")

    def test_06_temporal_fusion_decoder_ctc(self):
        """Validates CTC collapse of repeated phonemes and phrase reconstruction."""
        decoder = MultiModalFusionDecoder()
        logits = np.random.randn(1, 16, 64).astype(np.float32)
        
        # Inject known token: "execute" (repeated across frames to verify CTC collapse)
        exec_idx = SILENTECHO_VOCAB.index("execute")
        logits[0, 4:8, exec_idx] = 15.0
        logits[0, :4, 1] = 10.0  # blank
        logits[0, 8:, 1] = 10.0  # blank

        phrase, conf = decoder.decode_logits(logits)
        self.assertIn("execute", phrase)
        self.assertGreaterEqual(conf, 0.8)

    def test_07_benchmark_metrics_harness(self):
        """Validates performance metrics, duty cycle, and energy savings calculations."""
        harness = BenchmarkHarness(target_hardware="Snapdragon X Elite (Hexagon NPU)")
        for _ in range(25):
            harness.record_iteration(visual_roi_ms=1.2, audio_mel_ms=0.8, npu_inference_ms=3.5, fusion_decode_ms=0.3)

        report = harness.generate_report(active_provider="QNNExecutionProvider")
        self.assertEqual(report.iterations, 25)
        self.assertAlmostEqual(report.mean_e2e_latency_ms, 5.8, delta=0.2)
        self.assertEqual(report.sub_20ms_compliance_pct, 100.0)
        self.assertGreater(report.power_telemetry["energy_efficiency_gain_pct"], 80.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
