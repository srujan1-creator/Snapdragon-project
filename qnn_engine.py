"""
Project SilentEcho: Qualcomm Hexagon NPU Engine & ONNX Runtime Session Manager
Optimized for Snapdragon X Elite / X Plus on Windows 11 ARM64

Features:
- Primary acceleration: QNNExecutionProvider targeting Hexagon Tensor Processor (HTP v73/v75).
- Specific Qualcomm HTP configuration options:
  * backend_path: "QnnHtp.dll"
  * htp_performance_mode: "burst" (maximum NPU clock frequencies)
  * htp_graph_finalization_optimization_mode: "3" (full AOT VTCM optimization)
  * enable_htp_fp16_precision: "1" (mixed INT8/FP16 precision)
- Multi-tier zero-crash fallback hierarchy:
  1. QNNExecutionProvider (Qualcomm Hexagon NPU - 45 TOPS)
  2. DmlExecutionProvider (Qualcomm Adreno GPU / DirectML)
  3. CPUExecutionProvider (Qualcomm Oryon CPU / Host CPU)
- Built-in Autonomous Multimodal ONNX Model Synthesizer:
  Generates a dual-stream fusion ONNX model on the fly conforming to:
  * visual_frames: (1, 16, 1, 40, 80) float32
  * audio_mel:     (1, 80, 64) float32
  * logits:        (1, 16, 64) float32 (temporal phoneme/token sequence)
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SilentEcho.QNNEngine")

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    logger.warning("onnxruntime not installed. QNNEngine will run in simulated emulation mode.")

try:
    import onnx
    from onnx import helper, TensorProto
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@dataclass
class QNNConfig:
    """
    Qualcomm Hexagon NPU Configuration Parameters for ONNX Runtime QNN EP.
    Target: Snapdragon X Series (X1E-80-100, X1E-84-100, X1P-64-100).
    """
    backend_path: str = "QnnHtp.dll"                  # HTP runtime backend
    htp_performance_mode: str = "burst"               # "burst", "sustained_high_performance", "power_saver"
    htp_graph_finalization_optimization_mode: str = "3" # 3 = Max performance AOT optimization
    enable_htp_fp16_precision: str = "1"              # Allow FP16/INT8 mixed precision
    soc_model: str = "60"                             # Snapdragon X Elite architecture code
    vtcm_mb: str = "8"                                # Vector Tightly-Coupled Memory reservation (MB)
    profiling_level: str = "off"                      # "off", "basic", "detailed"
    preferred_provider: str = "auto"                  # "auto", "qnn", "dml", "cpu"
    intra_op_num_threads: int = 4
    inter_op_num_threads: int = 2


def generate_multimodal_dummy_onnx(model_path: str, visual_len: int = 16, mel_len: int = 64, vocab_size: int = 64):
    """
    Synthesizes a dual-stream multimodal ONNX model graph for Project SilentEcho.
    Allows end-to-end execution on Hexagon NPU / DirectML / CPU without external weights.
    
    Inputs:
        - visual_frames: (1, 16, 1, 40, 80) float32 [Lip ROI video sequence]
        - audio_mel:     (1, 80, 64) float32        [Log-Mel spectrogram]
    Output:
        - token_logits:  (1, 16, 64) float32        [Phoneme/vocabulary token logits]
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX library unavailable. Skipping ONNX model serialization.")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    logger.info("Synthesizing dual-stream multimodal ONNX graph: %s", model_path)

    # 1. Define Graph Inputs
    # Visual: [1, 16, 1, 40, 80]
    visual_input = helper.make_tensor_value_info('visual_frames', TensorProto.FLOAT, [1, visual_len, 1, 40, 80])
    # Audio: [1, 80, 64]
    audio_input = helper.make_tensor_value_info('audio_mel', TensorProto.FLOAT, [1, 80, mel_len])
    # Output: [1, 16, 64]
    output_logits = helper.make_tensor_value_info('token_logits', TensorProto.FLOAT, [1, visual_len, vocab_size])

    nodes = []
    initializers = []

    # Visual branch: Reshape (1, 16, 1, 40, 80) -> (16, 3200) -> MatMul with W_v (3200, 64) -> (16, 64) -> Reshape to (1, 16, 64)
    v_shape_tensor = helper.make_tensor('v_flat_shape', TensorProto.INT64, [2], [visual_len, 40 * 80])
    initializers.append(v_shape_tensor)
    nodes.append(helper.make_node('Reshape', ['visual_frames', 'v_flat_shape'], ['v_flattened']))

    # Random weights for visual projection
    np.random.seed(42)
    w_v_data = (np.random.randn(40 * 80, 64) * 0.02).astype(np.float32)
    w_v_tensor = helper.make_tensor('W_v', TensorProto.FLOAT, [40 * 80, 64], w_v_data.flatten().tolist())
    initializers.append(w_v_tensor)
    nodes.append(helper.make_node('MatMul', ['v_flattened', 'W_v'], ['v_projected'])) # (16, 64)

    # Audio branch: Transpose (1, 80, 64) -> (1, 64, 80) -> Reshape to (64, 80) -> MatMul with W_a (80, 64) -> (64, 64)
    # Then average pool across time or interpolate to (16, 64)
    a_perm = helper.make_node('Transpose', ['audio_mel'], ['audio_transposed'], perm=[0, 2, 1])
    nodes.append(a_perm)

    # Slice/Downsample audio temporal steps from 64 to 16 using simple Gather or MatMul
    # Let's project audio [1, 64, 80] flattened to [64, 80]
    a_shape_tensor = helper.make_tensor('a_flat_shape', TensorProto.INT64, [2], [mel_len, 80])
    initializers.append(a_shape_tensor)
    nodes.append(helper.make_node('Reshape', ['audio_transposed', 'a_flat_shape'], ['a_flattened']))

    w_a_data = (np.random.randn(80, 64) * 0.02).astype(np.float32)
    w_a_tensor = helper.make_tensor('W_a', TensorProto.FLOAT, [80, 64], w_a_data.flatten().tolist())
    initializers.append(w_a_tensor)
    nodes.append(helper.make_node('MatMul', ['a_flattened', 'W_a'], ['a_projected_64'])) # (64, 64)

    # Audio temporal downsampler: (16, 64) downsampler matrix
    downsample_matrix = np.zeros((visual_len, mel_len), dtype=np.float32)
    step = mel_len // visual_len
    for i in range(visual_len):
        downsample_matrix[i, i * step : (i + 1) * step] = 1.0 / step
    ds_tensor = helper.make_tensor('W_downsample', TensorProto.FLOAT, [visual_len, mel_len], downsample_matrix.flatten().tolist())
    initializers.append(ds_tensor)
    nodes.append(helper.make_node('MatMul', ['W_downsample', 'a_projected_64'], ['a_projected'])) # (16, 64)

    # Cross-Attention / Multimodal Fusion: Add visual and acoustic embeddings + Bias
    nodes.append(helper.make_node('Add', ['v_projected', 'a_projected'], ['fused_embeddings'])) # (16, 64)
    
    # LayerNorm / Activation (GELU approximation via FastGelu or Sigmoid*x or Relu)
    nodes.append(helper.make_node('Relu', ['fused_embeddings'], ['fused_activated']))

    # Output classifier projection: (16, 64) -> (16, vocab_size)
    w_out_data = (np.random.randn(64, vocab_size) * 0.05).astype(np.float32)
    w_out_tensor = helper.make_tensor('W_out', TensorProto.FLOAT, [64, vocab_size], w_out_data.flatten().tolist())
    initializers.append(w_out_tensor)
    nodes.append(helper.make_node('MatMul', ['fused_activated', 'W_out'], ['raw_logits'])) # (16, vocab_size)

    # Reshape back to batch: [1, 16, vocab_size]
    out_shape_tensor = helper.make_tensor('out_shape', TensorProto.INT64, [3], [1, visual_len, vocab_size])
    initializers.append(out_shape_tensor)
    nodes.append(helper.make_node('Reshape', ['raw_logits', 'out_shape'], ['token_logits']))

    # Build Graph & Model
    graph = helper.make_graph(
        nodes=nodes,
        name='SilentEchoDualStreamFusion',
        inputs=[visual_input, audio_input],
        outputs=[output_logits],
        initializer=initializers
    )

    model = helper.make_model(graph, producer_name='ProjectSilentEcho')
    model.opset_import[0].version = 17
    onnx.save(model, model_path)
    logger.info("Multimodal ONNX model successfully generated: %s (Size: %.2f KB)", model_path, os.path.getsize(model_path) / 1024.0)
    return True


class SilentEchoQNNEngine:
    """
    ONNX Runtime Inference Engine targeting Qualcomm Hexagon NPU (QNN EP)
    with seamless DirectML and CPU fallback.
    """
    def __init__(self, model_path: Optional[str] = None, config: Optional[QNNConfig] = None):
        self.config = config or QNNConfig()
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), "models", "silentecho_multimodal.onnx")
        self.session: Optional[ort.InferenceSession] = None
        self.active_provider: str = "NONE"
        self.provider_options_applied: Dict[str, Any] = {}
        self.input_names: List[str] = []
        self.output_names: List[str] = []

        self._initialize_engine()

    def _initialize_engine(self):
        # Ensure model exists
        if not os.path.exists(self.model_path):
            logger.info("ONNX model not found at %s. Synthesizing dual-stream multimodal graph...", self.model_path)
            generate_multimodal_dummy_onnx(self.model_path)

        if not ORT_AVAILABLE:
            self.active_provider = "SIMULATED_PYTHON"
            logger.warning("Running in simulated software fallback mode (no ONNX Runtime).")
            return

        available_providers = ort.get_available_providers()
        logger.info("Available ONNX Runtime Providers on this host: %s", available_providers)

        # Build session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = self.config.intra_op_num_threads
        sess_options.inter_op_num_threads = self.config.inter_op_num_threads

        # Determine provider list and options
        providers = []
        provider_options = []

        # 1. Qualcomm QNN Execution Provider (Hexagon NPU)
        qnn_options = {
            "backend_path": self.config.backend_path,
            "htp_performance_mode": self.config.htp_performance_mode,
            "htp_graph_finalization_optimization_mode": self.config.htp_graph_finalization_optimization_mode,
            "enable_htp_fp16_precision": self.config.enable_htp_fp16_precision,
            "soc_model": self.config.soc_model,
        }

        # Check preference
        if self.config.preferred_provider == "cpu":
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
        elif self.config.preferred_provider == "dml" and "DmlExecutionProvider" in available_providers:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{"device_id": 0}, {}]
        elif self.config.preferred_provider == "qnn" or self.config.preferred_provider == "auto":
            if "QNNExecutionProvider" in available_providers:
                providers.append("QNNExecutionProvider")
                provider_options.append(qnn_options)
            if "DmlExecutionProvider" in available_providers:
                providers.append("DmlExecutionProvider")
                provider_options.append({"device_id": 0})
            providers.append("CPUExecutionProvider")
            provider_options.append({})

        # Attempt session creation with graceful fallback
        session_created = False
        for idx, provider_name in enumerate(providers):
            try:
                logger.info("Attempting to load model with provider: %s", provider_name)
                opts = [provider_options[idx]] if idx < len(provider_options) else [{}]
                self.session = ort.InferenceSession(
                    self.model_path,
                    sess_options=sess_options,
                    providers=[provider_name],
                    provider_options=opts
                )
                self.active_provider = provider_name
                self.provider_options_applied = opts[0]
                session_created = True
                logger.info("Successfully loaded ONNX model using provider: %s", provider_name)
                break
            except Exception as e:
                logger.warning("Provider %s failed initialization: %s. Falling back to next available provider.", provider_name, e)

        if not session_created:
            logger.info("Attempting final fallback to default CPUExecutionProvider...")
            self.session = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
            self.active_provider = "CPUExecutionProvider"

        # Cache input/output names
        if self.session is not None:
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            logger.info("Model inputs: %s | Model outputs: %s", self.input_names, self.output_names)

    def infer(self, visual_frames: np.ndarray, audio_mel: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Executes multimodal inference on Hexagon NPU / Active Provider.
        Args:
            visual_frames: (1, 16, 1, 40, 80) float32
            audio_mel:     (1, 80, 64) float32
        Returns:
            logits: (1, 16, vocab_size) float32
            latency_ms: Execution duration in milliseconds
        """
        t_start = time.perf_counter()

        if self.session is not None:
            # Prepare feed dictionary matching model input names
            feed = {
                self.input_names[0]: visual_frames,
                self.input_names[1]: audio_mel
            }
            outputs = self.session.run(self.output_names, feed)
            logits = outputs[0]
        else:
            # High-fidelity mathematical emulation when ORT is unavailable
            if not hasattr(self, '_emu_w_v'):
                np.random.seed(42)
                self._emu_w_v = (np.random.randn(40 * 80, 64) * 0.02).astype(np.float32)
                self._emu_w_a = (np.random.randn(80, 64) * 0.02).astype(np.float32)
                self._emu_w_out = (np.random.randn(64, 64) * 0.05).astype(np.float32)
            
            v_flat = visual_frames.reshape(16, -1)
            v_proj = np.matmul(v_flat, self._emu_w_v)
            a_trans = np.squeeze(audio_mel, axis=0).T # (64, 80)
            a_ds = a_trans.reshape(16, 4, 80).mean(axis=1) # (16, 80)
            a_proj = np.matmul(a_ds, self._emu_w_a)
            fused = np.maximum(0, v_proj + a_proj)
            logits_2d = np.matmul(fused, self._emu_w_out)
            logits = np.expand_dims(logits_2d, axis=0).astype(np.float32)

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return logits, t_elapsed_ms

    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Returns runtime execution target metadata for telemetry and benchmarking.
        """
        return {
            "active_provider": self.active_provider,
            "target_silicon": "Qualcomm Hexagon NPU (HTP)" if "QNN" in self.active_provider else (
                "Qualcomm Adreno GPU (DirectML)" if "Dml" in self.active_provider else "Qualcomm Oryon / Host CPU"
            ),
            "htp_performance_mode": self.config.htp_performance_mode,
            "htp_graph_finalization_optimization_mode": self.config.htp_graph_finalization_optimization_mode,
            "model_path": self.model_path,
            "inputs": self.input_names,
            "outputs": self.output_names
        }


if __name__ == "__main__":
    print("=== Testing Project SilentEcho Qualcomm QNN Engine ===")
    
    # Test model generation & execution
    engine = SilentEchoQNNEngine()
    info = engine.get_hardware_info()
    print("Hardware Target Info:", info)

    # Synthesize dummy inputs
    dummy_visual = np.random.uniform(-1.0, 1.0, (1, 16, 1, 40, 80)).astype(np.float32)
    dummy_audio = np.random.uniform(-2.0, 2.0, (1, 80, 64)).astype(np.float32)

    print("\nWarming up NPU / Provider pipeline...")
    for _ in range(5):
        _, _ = engine.infer(dummy_visual, dummy_audio)

    print("Running 20 benchmark inference iterations...")
    latencies = []
    for i in range(20):
        logits, lat = engine.infer(dummy_visual, dummy_audio)
        latencies.append(lat)
        if (i + 1) % 5 == 0:
            print(f"Iteration {i+1:02d}: Latency = {lat:.2f} ms | Output Shape = {logits.shape}")

    avg_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    print(f"\n--- NPU Inference Results ({info['active_provider']}) ---")
    print(f"Mean Latency: {avg_lat:.2f} ms | 95th Percentile: {p95_lat:.2f} ms | FPS: {1000.0/avg_lat:.1f}")
