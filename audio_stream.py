"""
Project SilentEcho: Sub-Vocal Acoustic Audio Stream & Log-Mel Spectrogram Engine
Optimized for HP Poly Studio Microphone Array on Snapdragon X Elite / X Plus

Features:
- 16kHz mono non-blocking circular buffer audio capture.
- Sub-vocal acoustic conditioning:
  * 4th-order Butterworth bandpass filtering (120 Hz - 7200 Hz).
  * High-frequency whisper pre-emphasis (y[t] = x[t] - 0.97 * x[t-1]).
  * Automatic Gain Control (AGC) & dynamic range expansion for near-silent phonations.
- Zero-PyTorch real-time Log-Mel Spectrogram engine (80 Mel channels, 25ms window, 10ms hop).
- Formats acoustic tensors to (1, 80, T_mel) float32 for Hexagon NPU execution.
- Synthetic mock audio generator for headless CI/CD and systems without physical microphones.
"""

import sys
import time
import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SilentEcho.AudioStream")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False
    logger.warning("sounddevice unavailable or audio host API error. Falling back to synthetic audio stream.")


@dataclass
class AudioConfig:
    sample_rate: int = 16000          # Standard for Whisper/Conformer ASR
    channels: int = 1                 # Mono
    chunk_size: int = 320             # 20ms audio chunks (16000 * 0.02)
    buffer_seconds: float = 3.0       # Circular ring buffer capacity
    n_mels: int = 80                  # Whisper-standard Mel frequency channels
    n_fft: int = 512                  # FFT length (power of 2)
    win_length: int = 400             # 25ms analysis window (16000 * 0.025)
    hop_length: int = 160             # 10ms frame step (16000 * 0.010)
    f_min: float = 120.0              # Low cut (filter out HP laptop fan rumble)
    f_max: float = 7200.0             # High cut (Nyquist headroom)
    preemphasis: float = 0.97         # Whisper fricative enhancement factor
    agc_target_rms: float = 0.15      # Target RMS amplitude for whisper boost
    agc_max_gain: float = 24.0        # Max linear amplification gain (~27 dB)
    mel_window_frames: int = 64       # Number of temporal Mel frames fed to NPU (~640ms)


class MelFilterbank:
    """
    High-speed standalone NumPy Mel Filterbank.
    Converts raw audio waveform chunks into 80-bin Log-Mel Spectrogram features
    without requiring heavyweight PyTorch or librosa on edge devices.
    """
    def __init__(self, config: AudioConfig):
        self.config = config
        self.sr = config.sample_rate
        self.n_fft = config.n_fft
        self.n_mels = config.n_mels
        self.win_length = config.win_length
        self.hop_length = config.hop_length
        
        # Periodic Hann window
        self.window = np.hanning(self.win_length).astype(np.float32)
        # Construct Mel filterbank matrix
        self.mel_basis = self._create_mel_basis(
            sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
            fmin=config.f_min, fmax=config.f_max
        )

    def _hz_to_mel(self, hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _create_mel_basis(self, sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
        # Mel scale linearly spaced points
        min_mel = self._hz_to_mel(np.array([fmin]))[0]
        max_mel = self._hz_to_mel(np.array([fmax]))[0]
        mels = np.linspace(min_mel, max_mel, n_mels + 2)
        hzs = self._mel_to_hz(mels)

        # FFT bin frequencies
        fft_freqs = np.linspace(0.0, sr / 2.0, (n_fft // 2) + 1)
        weights = np.zeros((n_mels, (n_fft // 2) + 1), dtype=np.float32)

        for i in range(n_mels):
            lower = hzs[i]
            center = hzs[i + 1]
            upper = hzs[i + 2]

            # Up-slope
            up = (fft_freqs - lower) / (center - lower + 1e-8)
            # Down-slope
            down = (upper - fft_freqs) / (upper - center + 1e-8)
            weights[i] = np.maximum(0.0, np.minimum(up, down))

            # Slaney normalization
            enorm = 2.0 / (upper - lower + 1e-8)
            weights[i] *= enorm

        return weights

    def compute_log_mel(self, audio: np.ndarray) -> np.ndarray:
        """
        Computes 80-bin log-Mel spectrogram from 1D audio buffer.
        Returns:
            log_mel: (n_mels, T_frames) float32 array
        """
        if len(audio) < self.win_length:
            return np.zeros((self.n_mels, 1), dtype=np.float32)

        # Frame audio with sliding window
        n_frames = 1 + (len(audio) - self.win_length) // self.hop_length
        if n_frames <= 0:
            return np.zeros((self.n_mels, 1), dtype=np.float32)

        shape = (n_frames, self.win_length)
        strides = (audio.strides[0] * self.hop_length, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)

        # Windowing & Zero-padding to n_fft
        windowed = frames * self.window
        # Real FFT
        stft = np.fft.rfft(windowed, n=self.n_fft, axis=-1)
        magnitudes = np.abs(stft) ** 2  # Power spectrum: (n_frames, n_fft // 2 + 1)

        # Mel projection: (n_mels, n_frames)
        mel_spec = np.dot(self.mel_basis, magnitudes.T)
        
        # Log compression (clamped to prevent -inf)
        log_mel = np.log(np.maximum(mel_spec, 1e-5))
        # Standard Whisper/Speech normalization
        log_mel = (log_mel + 4.0) / 4.0
        return log_mel.astype(np.float32)


class SubVocalConditioner:
    """
    Acoustic front-end specialized for sub-vocal whispers captured by HP Poly Studio mic array.
    Applies bandpass filtering, high-frequency whisper pre-emphasis, and automatic gain expansion.
    """
    def __init__(self, config: AudioConfig):
        self.config = config
        # Design 4th-order Butterworth bandpass filter
        nyquist = config.sample_rate / 2.0
        low = config.f_min / nyquist
        high = min(0.99, config.f_max / nyquist)
        self.b, self.a = signal.butter(4, [low, high], btype='band')
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.last_sample = 0.0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """
        Processes a raw PCM audio chunk:
        1. Bandpass filter (120 Hz - 7200 Hz) to eliminate fan rumble.
        2. High-frequency whisper pre-emphasis (y[t] = x[t] - 0.97*x[t-1]).
        3. Dynamic gain control (boosts sub-vocal murmurs to perceptible levels).
        """
        # 1. Bandpass filter with persistent filter state
        filtered, self.zi = signal.lfilter(self.b, self.a, chunk, zi=self.zi * self.last_sample if self.zi is not None else None)
        
        # 2. Whisper pre-emphasis (emphasizes unvoiced fricatives)
        preemph = np.empty_like(filtered)
        preemph[0] = filtered[0] - self.config.preemphasis * self.last_sample
        preemph[1:] = filtered[1:] - self.config.preemphasis * filtered[:-1]
        self.last_sample = filtered[-1] if len(filtered) > 0 else 0.0

        # 3. Dynamic AGC (Adaptive Gain Control for whispers)
        rms = np.sqrt(np.mean(preemph ** 2) + 1e-9)
        if rms < self.config.agc_target_rms:
            gain = min(self.config.agc_max_gain, self.config.agc_target_rms / (rms + 1e-4))
        else:
            gain = 1.0

        conditioned = np.clip(preemph * gain, -1.0, 1.0)
        return conditioned.astype(np.float32)


class AudioStream:
    """
    High-performance circular buffer audio capture stream using sounddevice.
    Designed for the HP Poly Studio onboard microphone array.
    """
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.buffer_size = int(self.config.sample_rate * self.config.buffer_seconds)
        self.ring_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_idx = 0
        self.total_samples_written = 0
        self.lock = threading.Lock()
        
        self.conditioner = SubVocalConditioner(self.config)
        self.mel_engine = MelFilterbank(self.config)
        self.stream: Optional[sd.InputStream] = None
        self.running = False

    def start(self) -> bool:
        if not SOUNDDEVICE_AVAILABLE:
            logger.warning("sounddevice unavailable. Running in simulated audio mode.")
            return False

        try:
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.chunk_size,
                dtype='float32',
                callback=self._audio_callback
            )
            self.stream.start()
            self.running = True
            logger.info("HP Poly Studio audio stream active (16kHz mono, %d sample blocks).", self.config.chunk_size)
            return True
        except Exception as e:
            logger.warning("Failed to open physical audio device: %s. Falling back to synthetic stream.", e)
            return False

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.debug("Audio stream status flag: %s", status)

        pcm = indata[:, 0]
        # Condition sub-vocal whispers
        processed = self.conditioner.process(pcm)

        with self.lock:
            n = len(processed)
            end_idx = self.write_idx + n
            if end_idx <= self.buffer_size:
                self.ring_buffer[self.write_idx:end_idx] = processed
            else:
                part1 = self.buffer_size - self.write_idx
                part2 = n - part1
                self.ring_buffer[self.write_idx:] = processed[:part1]
                self.ring_buffer[:part2] = processed[part1:]
            self.write_idx = (self.write_idx + n) % self.buffer_size
            self.total_samples_written += n

    def get_latest_audio(self, num_samples: int) -> np.ndarray:
        """
        Retrieves the most recent num_samples from the circular buffer in temporal order.
        """
        with self.lock:
            num = min(num_samples, self.buffer_size)
            curr = self.write_idx
            if curr >= num:
                return self.ring_buffer[curr - num:curr].copy()
            else:
                part2 = self.ring_buffer[:curr]
                part1 = self.ring_buffer[self.buffer_size - (num - curr):]
                return np.concatenate((part1, part2))

    def get_latest_mel_tensor(self, num_frames: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Computes and formats Log-Mel Spectrogram for Hexagon NPU:
        Shape: (1, n_mels, T_mel) float32, e.g. (1, 80, 64)
        """
        frames_needed = num_frames or self.config.mel_window_frames
        # Required audio samples = (frames_needed - 1) * hop_length + win_length
        samples_needed = (frames_needed - 1) * self.config.hop_length + self.config.win_length

        if self.total_samples_written < samples_needed:
            return None

        audio_segment = self.get_latest_audio(samples_needed)
        mel = self.mel_engine.compute_log_mel(audio_segment) # (80, T)
        
        # Ensure exact frame width
        if mel.shape[1] < frames_needed:
            pad = np.zeros((self.config.n_mels, frames_needed - mel.shape[1]), dtype=np.float32)
            mel = np.hstack((mel, pad))
        elif mel.shape[1] > frames_needed:
            mel = mel[:, -frames_needed:]

        # Add batch dimension -> (1, n_mels, T_mel)
        tensor = np.expand_dims(mel, axis=0)
        return tensor.astype(np.float32)

    def stop(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        logger.info("Audio stream closed.")


class MockAudioStream:
    """
    High-fidelity synthetic whisper audio generator for testing, headless CI/CD,
    and environments without physical audio hardware.
    Generates realistic unvoiced turbulent acoustic noise modulated by formant resonances.
    """
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.buffer_size = int(self.config.sample_rate * self.config.buffer_seconds)
        self.ring_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_idx = 0
        self.total_samples_written = 0
        self.lock = threading.Lock()
        
        self.conditioner = SubVocalConditioner(self.config)
        self.mel_engine = MelFilterbank(self.config)
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        self.running = True
        self.thread = threading.Thread(target=self._generator_loop, daemon=True, name="SilentEcho-MockAudio")
        self.thread.start()
        logger.info("Mock sub-vocal audio stream active (16kHz mono synthetic).")
        return True

    def _generator_loop(self):
        chunk_len = self.config.chunk_size
        dt = chunk_len / self.config.sample_rate
        t = 0.0

        while self.running:
            start_tick = time.time()
            t += dt

            # Synthesize unvoiced turbulent breath/whisper acoustics
            # Formant resonance frequencies typical of English silent phonemes (/sh/, /s/, /t/)
            f_formant1 = 1800.0 + 400.0 * np.sin(2.0 * np.pi * 1.2 * t)
            f_formant2 = 3200.0 + 600.0 * np.cos(2.0 * np.pi * 0.9 * t)

            # White noise base (turbulent airflow)
            raw_noise = np.random.normal(0, 0.04, chunk_len).astype(np.float32)
            
            # Modulate with formant resonant filters
            time_axis = np.linspace(t, t + dt, chunk_len, endpoint=False)
            formant_mod = 0.5 * np.sin(2.0 * np.pi * f_formant1 * time_axis) + 0.3 * np.sin(2.0 * np.pi * f_formant2 * time_axis)
            whisper_pcm = raw_noise * (1.0 + 0.6 * formant_mod.astype(np.float32))

            # Condition through sub-vocal pipeline
            processed = self.conditioner.process(whisper_pcm)

            with self.lock:
                n = len(processed)
                end_idx = self.write_idx + n
                if end_idx <= self.buffer_size:
                    self.ring_buffer[self.write_idx:end_idx] = processed
                else:
                    p1 = self.buffer_size - self.write_idx
                    p2 = n - p1
                    self.ring_buffer[self.write_idx:] = processed[:p1]
                    self.ring_buffer[:p2] = processed[p1:]
                self.write_idx = (self.write_idx + n) % self.buffer_size
                self.total_samples_written += n

            elapsed = time.time() - start_tick
            sleep_time = max(0.001, dt - elapsed)
            time.sleep(sleep_time)

    def get_latest_audio(self, num_samples: int) -> np.ndarray:
        with self.lock:
            num = min(num_samples, self.buffer_size)
            curr = self.write_idx
            if curr >= num:
                return self.ring_buffer[curr - num:curr].copy()
            else:
                p2 = self.ring_buffer[:curr]
                p1 = self.ring_buffer[self.buffer_size - (num - curr):]
                return np.concatenate((p1, p2))

    def get_latest_mel_tensor(self, num_frames: Optional[int] = None) -> Optional[np.ndarray]:
        frames_needed = num_frames or self.config.mel_window_frames
        samples_needed = (frames_needed - 1) * self.config.hop_length + self.config.win_length

        if self.total_samples_written < samples_needed:
            return None

        audio_segment = self.get_latest_audio(samples_needed)
        mel = self.mel_engine.compute_log_mel(audio_segment)

        if mel.shape[1] < frames_needed:
            pad = np.zeros((self.config.n_mels, frames_needed - mel.shape[1]), dtype=np.float32)
            mel = np.hstack((mel, pad))
        elif mel.shape[1] > frames_needed:
            mel = mel[:, -frames_needed:]

        tensor = np.expand_dims(mel, axis=0)
        return tensor.astype(np.float32)

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Mock audio stream stopped.")


def create_audio_stream(config: Optional[AudioConfig] = None, force_mock: bool = False):
    """
    Factory function: returns AudioStream if physical mic available, else MockAudioStream.
    """
    if force_mock:
        stream = MockAudioStream(config)
        stream.start()
        return stream

    stream = AudioStream(config)
    success = stream.start()
    if not success:
        logger.info("Falling back to synthetic MockAudioStream.")
        stream = MockAudioStream(config)
        stream.start()
    return stream


if __name__ == "__main__":
    print("=== Testing Project SilentEcho Sub-Vocal Audio Stream ===")
    cfg = AudioConfig(mel_window_frames=64)
    use_mock = "--mock" in sys.argv
    audio = create_audio_stream(cfg, force_mock=use_mock)

    try:
        print("Buffering sub-vocal acoustic stream...")
        time.sleep(1.0)
        for i in range(10):
            mel_tensor = audio.get_latest_mel_tensor()
            if mel_tensor is not None:
                print(f"[{i+1}/10] Extracted Log-Mel tensor: shape={mel_tensor.shape}, dtype={mel_tensor.dtype}, min={mel_tensor.min():.2f}, max={mel_tensor.max():.2f}")
            else:
                print(f"[{i+1}/10] Audio buffer filling ({audio.total_samples_written} samples)...")
            time.sleep(0.1)
    finally:
        audio.stop()
        print("Audio test completed successfully.")
