"""MERT audio embedding extraction.

Wraps the MERT-v1-330M model for computing 768-dim audio embeddings
from 30-second MP3 preview clips.
"""

import io
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import librosa
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from config import MERT_MODEL_ID, MERT_SAMPLE_RATE, MERT_EMBEDDING_DIM

logger = logging.getLogger(__name__)


class MERTEmbedder:
    """Compute MERT audio embeddings from raw MP3 bytes."""

    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info("Loading MERT model %s on %s...", MERT_MODEL_ID, device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL_ID)
        self.model = AutoModel.from_pretrained(MERT_MODEL_ID, trust_remote_code=True)
        self.model.to(device).eval()
        logger.info("MERT model loaded.")

    def _decode_audio(self, mp3_bytes):
        """Decode MP3 bytes to a waveform at MERT_SAMPLE_RATE.

        Returns numpy array of shape (samples,) or None on failure.

        Uses a temporary file because librosa's BytesIO path requires ffmpeg,
        which is not always available (e.g. macOS without Homebrew). A named
        temp file lets the OS audio backend (macCA on macOS, ffmpeg on Linux)
        handle format detection. Overhead is negligible vs. MERT inference.
        """
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(mp3_bytes)
                tmp_path = f.name
            waveform, _ = librosa.load(tmp_path, sr=MERT_SAMPLE_RATE, mono=True)
            return waveform
        except Exception as e:
            logger.warning("Failed to decode audio: %s", e)
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _pad_or_truncate(self, waveform, target_length):
        """Pad with zeros or truncate waveform to exact target_length."""
        if len(waveform) >= target_length:
            return waveform[:target_length]
        padded = np.zeros(target_length, dtype=np.float32)
        padded[: len(waveform)] = waveform
        return padded

    def embed(self, mp3_bytes):
        """Compute a 768-dim embedding from MP3 bytes.

        Returns L2-normalized numpy float32 array of shape (768,), or None.
        """
        waveform = self._decode_audio(mp3_bytes)
        if waveform is None:
            return None

        inputs = self.processor(
            waveform, sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=False)

        # Mean-pool last hidden state over time dimension
        last_hidden = outputs.last_hidden_state  # (1, time, 768)
        embedding = last_hidden.mean(dim=1).squeeze(0)  # (768,)
        embedding = embedding.cpu().numpy().astype(np.float32)

        # L2-normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding

    def embed_batch(self, mp3_bytes_list):
        """Compute embeddings for a batch of MP3 clips.

        Returns list of numpy arrays (768,) — None entries for failed decodes.
        """
        # 30 seconds at MERT sample rate
        target_length = 30 * MERT_SAMPLE_RATE

        def _decode_one(mp3_bytes):
            waveform = self._decode_audio(mp3_bytes)
            if waveform is not None:
                waveform = self._pad_or_truncate(waveform, target_length)
            return waveform

        # Parallel CPU decode — overlap I/O and librosa work across the batch
        with ThreadPoolExecutor(max_workers=len(mp3_bytes_list)) as pool:
            decoded = list(pool.map(_decode_one, mp3_bytes_list))

        waveforms = []
        valid_indices = []
        for i, waveform in enumerate(decoded):
            if waveform is not None:
                waveforms.append(waveform)
                valid_indices.append(i)

        results = [None] * len(mp3_bytes_list)
        if not waveforms:
            return results

        inputs = self.processor(
            waveforms, sampling_rate=MERT_SAMPLE_RATE,
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        device_type = "cuda" if self.device == "cuda" else "cpu"
        with torch.no_grad(), torch.autocast(
            device_type=device_type, dtype=torch.float16, enabled=(device_type == "cuda")
        ):
            outputs = self.model(**inputs, output_hidden_states=False)

        last_hidden = outputs.last_hidden_state.float()  # (batch, time, 768)
        embeddings = last_hidden.mean(dim=1)  # (batch, 768)
        embeddings = embeddings.cpu().numpy().astype(np.float32)

        # L2-normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embeddings /= norms

        for batch_idx, orig_idx in enumerate(valid_indices):
            results[orig_idx] = embeddings[batch_idx]

        return results
