from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from fusion_simulation.core.rng import numpy_to_native

logger = logging.getLogger(__name__)

# Encoding tags shared between server and client so data_payload is decodable.
ENC_JSON_NATIVE = "json_native"
ENC_RAW_RGB = "raw_rgb"
ENC_RAW_DEPTH = "raw_depth"
ENC_EMPTY = "empty"


def encode_sensor_payload(sdata: dict[str, Any]) -> tuple[bytes, str, int, int, int]:
    # Encode sensor observation into a binary payload + metadata. Camera arrays
    # go raw (no json.dumps(default=str) text bloat that exceeded gRPC 4MB);
    # small structs use json with numpy->native conversion. Returns
    # (payload, encoding, width, height, channels).
    if not sdata:
        return b"", ENC_EMPTY, 0, 0, 0
    rgb = sdata.get("rgb")
    if rgb is None:
        rgb = sdata.get("image")
    if rgb is not None:
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            h, w = arr.shape
            ch = 1
        elif arr.ndim == 3:
            h, w, ch = arr.shape
        else:
            logger.warning("encode_sensor_payload: rgb ndim=%d unexpected, falling back to json", arr.ndim)
            return _json_native(sdata), ENC_JSON_NATIVE, 0, 0, 0
        return arr.tobytes(), ENC_RAW_RGB, w, h, ch
    depth = sdata.get("depth")
    if depth is not None:
        arr = np.asarray(depth, dtype=np.float32)
        if arr.ndim != 2:
            logger.warning("encode_sensor_payload: depth ndim=%d unexpected, falling back to json", arr.ndim)
            return _json_native(sdata), ENC_JSON_NATIVE, 0, 0, 0
        h, w = arr.shape
        return arr.tobytes(), ENC_RAW_DEPTH, w, h, 1
    return _json_native(sdata), ENC_JSON_NATIVE, 0, 0, 0


def encode_state_payload(state: dict[str, Any]) -> tuple[bytes, str]:
    # Sim-state is a small structured dict — encode as json with numpy->native
    # conversion so arrays serialize as lists, not str([[1 2 3]]).
    if not state:
        return b"{}", ENC_JSON_NATIVE
    return _json_native(state), ENC_JSON_NATIVE


def _json_native(data: dict[str, Any]) -> bytes:
    try:
        return json.dumps(numpy_to_native(data), ensure_ascii=False).encode("utf-8")
    except Exception as e:
        logger.error("json_native encode failed: %s", e)
        return b"{}"


def decode_sensor_payload(payload: bytes, encoding: str, width: int, height: int, channels: int) -> dict[str, Any]:
    # Client-side decode mirror of encode_sensor_payload.
    if encoding == ENC_EMPTY or not payload:
        return {}
    if encoding == ENC_RAW_RGB:
        arr = np.frombuffer(payload, dtype=np.uint8)
        if channels == 1:
            return {"rgb": arr.reshape(height, width)}
        return {"rgb": arr.reshape(height, width, channels)}
    if encoding == ENC_RAW_DEPTH:
        arr = np.frombuffer(payload, dtype=np.float32)
        return {"depth": arr.reshape(height, width)}
    if encoding == ENC_JSON_NATIVE:
        try:
            return json.loads(payload.decode("utf-8"))
        except Exception as e:
            logger.error("json_native decode failed: %s", e)
            return {}
    logger.warning("decode_sensor_payload: unknown encoding=%s", encoding)
    return {}
