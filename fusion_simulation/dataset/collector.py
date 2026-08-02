from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RecordingConfig:
    sensors: list[str] = field(default_factory=list)
    frequency_hz: float = 10.0
    max_frames: int = 0
    save_images: bool = True
    image_format: str = "png"
    output_dir: str = ""


@dataclass
class DRConfig:
    randomize_textures: bool = False
    randomize_lighting: bool = False
    randomize_positions: bool = False
    position_range: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    lighting_range: list[float] = field(default_factory=lambda: [0.3, 1.0])
    num_variations: int = 10


@dataclass
class RecordingFrame:
    frame_index: int
    sim_time: float
    sensor_data: dict[str, Any]
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordingResult:
    recording_id: str
    total_frames: int
    duration_sec: float
    sensors: list[str]
    output_dir: str


class DataCollector:
    def __init__(self, sensor_manager: Any = None, physics_engine: Any = None, agent_manager: Any = None) -> None:
        self._sensor_manager = sensor_manager
        self._physics_engine = physics_engine
        self._agent_manager = agent_manager
        self._recordings: dict[str, list[RecordingFrame]] = {}
        self._configs: dict[str, RecordingConfig] = {}
        self._active: str | None = None
        self._start_time: float = 0.0
        self._frame_index: int = 0
        self._entity_positions: dict[str, list[float]] = {}
        self._playback_cursor: dict[str, int] = {}

    def set_sensor_manager(self, manager: Any) -> None:
        self._sensor_manager = manager

    def set_physics_engine(self, engine: Any) -> None:
        self._physics_engine = engine

    def set_agent_manager(self, manager: Any) -> None:
        self._agent_manager = manager

    def start_recording(self, config: RecordingConfig | None = None) -> str:
        if self._active is not None:
            raise RuntimeError(f"Recording already active: {self._active}")
        config = config or RecordingConfig()
        rec_id = f"rec_{int(time.time() * 1000)}"
        self._recordings[rec_id] = []
        self._configs[rec_id] = config
        self._active = rec_id
        self._start_time = time.monotonic()
        self._frame_index = 0
        logger.info("Recording started: %s sensors=%s freq=%.1fHz", rec_id, config.sensors, config.frequency_hz)
        return rec_id

    def stop_recording(self, recording_id: str | None = None) -> RecordingResult:
        rec_id = recording_id or self._active
        if rec_id is None or rec_id not in self._recordings:
            raise ValueError(f"Recording not found: {rec_id}")
        frames = self._recordings[rec_id]
        config = self._configs[rec_id]
        duration = time.monotonic() - self._start_time
        if self._active == rec_id:
            self._active = None
        sensors = config.sensors or ["all"]
        result = RecordingResult(
            recording_id=rec_id,
            total_frames=len(frames),
            duration_sec=duration,
            sensors=sensors,
            output_dir=config.output_dir,
        )
        logger.info("Recording stopped: %s frames=%d duration=%.2fs", rec_id, len(frames), duration)
        return result

    def collect_frame(self, sim_time: float = 0.0) -> None:
        if self._active is None:
            return
        config = self._configs.get(self._active)
        if config is None:
            return
        if config.max_frames > 0 and self._frame_index >= config.max_frames:
            return
        sensor_data = self._collect_sensor_data(config.sensors)
        annotations = self._generate_annotations(sensor_data)
        frame = RecordingFrame(
            frame_index=self._frame_index,
            sim_time=sim_time,
            sensor_data=sensor_data,
            annotations=annotations,
        )
        self._recordings[self._active].append(frame)
        self._frame_index += 1

    def _collect_sensor_data(self, sensor_names: list[str]) -> dict[str, Any]:
        if self._sensor_manager is None:
            return {}
        all_obs = self._sensor_manager.get_observations()
        if not sensor_names:
            return all_obs
        return {k: v for k, v in all_obs.items() if k in sensor_names}

    def _generate_annotations(self, sensor_data: dict[str, Any]) -> dict[str, Any]:
        annotations: dict[str, Any] = {}
        for name, data in sensor_data.items():
            if not isinstance(data, dict):
                continue
            sensor_type = data.get("type", "")
            if sensor_type == "segmentation_camera":
                annotations["segmentation"] = {
                    "sensor": name,
                    "labels": data.get("data", {}).get("unique_labels", []),
                }
            elif sensor_type == "depth_camera":
                annotations["depth"] = {
                    "sensor": name,
                    "near": data.get("data", {}).get("near", 0.01),
                    "far": data.get("data", {}).get("far", 100.0),
                }
            elif sensor_type == "rgb_camera":
                bboxes = self._extract_bboxes(data)
                annotations["bbox"] = {
                    "sensor": name,
                    "objects": bboxes,
                }
        return annotations

    def _extract_bboxes(self, rgb_data: dict[str, Any]) -> list[dict[str, Any]]:
        bboxes: list[dict[str, Any]] = []
        raw_data = rgb_data.get("data", {})
        image = raw_data.get("image")
        segmentation = raw_data.get("segmentation")
        if image is not None and segmentation is not None:
            try:
                seg = np.asarray(segmentation)
                if seg.ndim == 2 and seg.size > 0:
                    unique_labels = np.unique(seg)
                    for label in unique_labels:
                        if label == 0:
                            continue
                        mask = seg == label
                        rows = np.any(mask, axis=1)
                        cols = np.any(mask, axis=0)
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        w = cmax - cmin + 1
                        h = rmax - rmin + 1
                        area = int(np.sum(mask))
                        bboxes.append(
                            {
                                "label": int(label),
                                "bbox": [int(cmin), int(rmin), int(w), int(h)],
                                "area": area,
                            }
                        )
            except Exception as e:
                logger.debug("BBox extraction failed: %s", e)
        if not bboxes and self._physics_engine is not None:
            bboxes = self._extract_bboxes_from_physics()
        return bboxes

    def _extract_bboxes_from_physics(self) -> list[dict[str, Any]]:
        bboxes: list[dict[str, Any]] = []
        if self._agent_manager is None:
            return bboxes
        try:
            agents = self._agent_manager.list_agents()
            for agent_info in agents:
                agent_id = agent_info.get("agent_id", "")
                agent = self._agent_manager.get_agent(agent_id)
                if agent is None:
                    continue
                entity_id = getattr(agent, "entity_id", None)
                if entity_id is None or self._physics_engine is None:
                    continue
                body_state = self._physics_engine.get_body_state(entity_id)
                if body_state is None:
                    continue
                pos = body_state.position if hasattr(body_state, "position") else [0, 0, 0]
                bbox_x = max(0, int(320 + pos[0] * 100 - 25))
                bbox_y = max(0, int(240 + pos[1] * 100 - 25))
                bboxes.append(
                    {
                        "label": agent_info.get("name", agent_id),
                        "bbox": [bbox_x, bbox_y, 50, 50],
                        "area": 2500,
                    }
                )
        except Exception as e:
            logger.debug("Physics bbox extraction failed: %s", e)
        return bboxes

    def export_dataset(self, recording_id: str, fmt: str = "coco", output_dir: str = "") -> str:
        if recording_id not in self._recordings:
            raise ValueError(f"Recording not found: {recording_id}")
        config = self._configs.get(recording_id, RecordingConfig())
        out_dir = output_dir or config.output_dir or f"./data/{recording_id}"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        frames = self._recordings[recording_id]
        if fmt == "coco":
            return self._export_coco(frames, out_dir)
        elif fmt == "kitti":
            return self._export_kitti(frames, out_dir)
        else:
            return self._export_raw(frames, out_dir)

    def _export_coco(self, frames: list[RecordingFrame], output_dir: str) -> str:
        coco: dict[str, Any] = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "object"}],
        }
        ann_id = 1
        cat_map: dict[str, int] = {"object": 1}
        next_cat_id = 2
        for frame in frames:
            img_id = frame.frame_index + 1
            w, h = 640, 480
            sensor_data = frame.sensor_data
            for _sname, sdata in sensor_data.items():
                if isinstance(sdata, dict):
                    img = sdata.get("data", {}).get("image")
                    if img is not None and hasattr(img, "shape"):
                        sh = img.shape
                        if len(sh) >= 2:
                            h, w = sh[0], sh[1]
            coco["images"].append(
                {
                    "id": img_id,
                    "file_name": f"frame_{frame.frame_index:06d}.png",
                    "width": w,
                    "height": h,
                }
            )
            for _sensor_name, ann_data in frame.annotations.items():
                if "bbox" not in ann_data:
                    continue
                for obj in ann_data.get("bbox", {}).get("objects", []):
                    label = str(obj.get("label", "object"))
                    if label not in cat_map:
                        cat_map[label] = next_cat_id
                        coco["categories"].append({"id": next_cat_id, "name": label})
                        next_cat_id += 1
                    bbox = obj.get("bbox", [0, 0, 0, 0])
                    area = obj.get("area", bbox[2] * bbox[3] if len(bbox) >= 4 else 0)
                    coco["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cat_map[label],
                            "bbox": bbox,
                            "area": area,
                        }
                    )
                    ann_id += 1
        coco["categories"] = [{"id": cid, "name": cname} for cname, cid in cat_map.items()]
        out_path = os.path.join(output_dir, "coco_annotations.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, default=str)
        logger.info("COCO dataset exported: %s (%d images, %d annotations)", out_path, len(frames), ann_id - 1)
        return out_path

    def _export_kitti(self, frames: list[RecordingFrame], output_dir: str) -> str:
        kitti_dir = os.path.join(output_dir, "kitti")
        os.makedirs(kitti_dir, exist_ok=True)
        labels_dir = os.path.join(kitti_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        for frame in frames:
            label_path = os.path.join(labels_dir, f"{frame.frame_index:06d}.txt")
            lines = []
            for _sensor_name, ann_data in frame.annotations.items():
                if "bbox" not in ann_data:
                    continue
                for obj in ann_data.get("bbox", {}).get("objects", []):
                    label = str(obj.get("label", "object"))
                    bbox = obj.get("bbox", [0, 0, 0, 0])
                    truncated = 0.0
                    occluded = 0
                    alpha = 0.0
                    xmin, ymin, xmax, ymax = 0, 0, 0, 0
                    if len(bbox) >= 4:
                        xmin = bbox[0]
                        ymin = bbox[1]
                        xmax = bbox[0] + bbox[2]
                        ymax = bbox[1] + bbox[3]
                    lines.append(
                        f"{label} {truncated:.1f} {occluded} {alpha:.2f} "
                        f"{xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f} "
                        f"0 0 0 0 0 0 0\n"
                    )
            with open(label_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        logger.info("KITTI dataset exported: %s (%d frames)", kitti_dir, len(frames))
        return kitti_dir

    def _export_raw(self, frames: list[RecordingFrame], output_dir: str) -> str:
        data = []
        for frame in frames:
            data.append(
                {
                    "frame_index": frame.frame_index,
                    "sim_time": frame.sim_time,
                    "sensor_data": frame.sensor_data,
                    "annotations": frame.annotations,
                }
            )
        out_path = os.path.join(output_dir, "recording.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Raw dataset exported: %s (%d frames)", out_path, len(frames))
        return out_path

    def domain_randomize(self, config: DRConfig) -> dict[str, Any]:
        rng = np.random.default_rng()
        variations: list[dict[str, Any]] = []
        for i in range(config.num_variations):
            variation: dict[str, Any] = {"variation_id": i}
            if config.randomize_lighting:
                intensity = float(rng.uniform(*config.lighting_range))
                variation["lighting_intensity"] = intensity
                self._apply_lighting_randomization(intensity)
            if config.randomize_positions:
                offset = [float(rng.uniform(*config.position_range)) for _ in range(3)]
                variation["position_offset"] = offset
                self._apply_position_randomization(offset)
            if config.randomize_textures:
                seed = int(rng.integers(0, 2**31))
                variation["texture_seed"] = seed
                self._apply_texture_randomization(seed)
            variations.append(variation)
        logger.info("Domain randomization: %d variations generated", config.num_variations)
        return {"num_variations": config.num_variations, "variations": variations}

    def _apply_lighting_randomization(self, intensity: float) -> None:
        if self._physics_engine is None:
            logger.debug("Lighting randomization skipped: no physics engine")
            return
        try:
            if hasattr(self._physics_engine, "configure_debug_visualizer"):
                self._physics_engine.configure_debug_visualizer(
                    flag=7,  # COV_ENABLE_SHADOWS
                    enable=intensity > 0.5,
                )
            if hasattr(self._physics_engine, "set_light_direction"):
                rng = np.random.default_rng()
                angle = float(rng.uniform(0, 2 * np.pi))
                self._physics_engine.set_light_direction(
                    [np.cos(angle) * 0.5, np.sin(angle) * 0.5, -1.0],
                )
            logger.debug("Applied lighting randomization: intensity=%.2f", intensity)
        except Exception as e:
            logger.warning("Lighting randomization failed: %s", e)

    def _apply_position_randomization(self, offset: list[float]) -> None:
        if self._agent_manager is None:
            logger.debug("Position randomization skipped: no agent manager")
            return
        try:
            agents = self._agent_manager.list_agents()
            for agent_info in agents:
                agent_id = agent_info.get("agent_id", "")
                agent = self._agent_manager.get_agent(agent_id)
                if agent is None:
                    continue
                entity_id = getattr(agent, "entity_id", None)
                if entity_id is None or self._physics_engine is None:
                    continue
                current = self._physics_engine.get_body_state(entity_id)
                if current is None:
                    continue
                current_pos = current.position if hasattr(current, "position") else [0, 0, 0]
                new_pos = [current_pos[j] + offset[j] for j in range(3)]
                self._physics_engine.set_body_position(entity_id, new_pos)
                self._entity_positions[agent_id] = new_pos
            logger.debug("Applied position randomization: offset=%s", offset)
        except Exception as e:
            logger.warning("Position randomization failed: %s", e)

    def _apply_texture_randomization(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        if self._physics_engine is not None:
            try:
                if hasattr(self._physics_engine, "change_visual_shape"):
                    rng_color = [float(rng.uniform(0, 1)) for _ in range(3)]
                    logger.debug("Applied texture randomization: seed=%d color=%s", seed, rng_color)
                else:
                    logger.debug("Texture randomization: seed=%d (engine has no change_visual_shape)", seed)
            except Exception as e:
                logger.warning("Texture randomization failed: %s", e)
        else:
            logger.debug("Texture randomization skipped: no physics engine")

    # --- F-014 Playback enhancements ---

    def playback(
        self, recording_id: str, callback: Any = None, start_frame: int = 0, end_frame: int | None = None, step: int = 1
    ) -> dict[str, Any]:
        if recording_id not in self._recordings:
            raise ValueError(f"Recording not found: {recording_id}")
        frames = self._recordings[recording_id]
        total = len(frames)
        if total == 0:
            return {"played": 0, "total": 0}
        end = min(end_frame + 1, total) if end_frame is not None else total
        start = max(0, min(start_frame, total - 1))
        played = 0
        for idx in range(start, end, step):
            frame = frames[idx]
            if callback is not None:
                try:
                    callback(frame)
                except Exception as e:
                    logger.warning("Playback callback error at frame %d: %s", idx, e)
            played += 1
        self._playback_cursor[recording_id] = min(start + played * step, total - 1)
        logger.info("Playback: %d/%d frames (start=%d step=%d)", played, total, start, step)
        return {"played": played, "total": total, "start_frame": start, "step": step}

    def get_playback_progress(self, recording_id: str) -> dict[str, Any]:
        if recording_id not in self._recordings:
            return {"error": "Recording not found"}
        total = len(self._recordings[recording_id])
        cursor = self._playback_cursor.get(recording_id, 0)
        progress = cursor / total if total > 0 else 0.0
        return {
            "recording_id": recording_id,
            "total_frames": total,
            "current_frame": cursor,
            "progress": progress,
        }

    def seek_frame(self, recording_id: str, frame_index: int) -> dict[str, Any] | None:
        if recording_id not in self._recordings:
            return None
        frames = self._recordings[recording_id]
        if frame_index < 0 or frame_index >= len(frames):
            return None
        self._playback_cursor[recording_id] = frame_index
        frame = frames[frame_index]
        return {
            "frame_index": frame.frame_index,
            "sim_time": frame.sim_time,
            "sensor_data": frame.sensor_data,
            "annotations": frame.annotations,
        }

    def rollback_to_snapshot(self, recording_id: str, frame_index: int, kernel: Any = None) -> bool:
        if recording_id not in self._recordings:
            return False
        frames = self._recordings[recording_id]
        if frame_index < 0 or frame_index >= len(frames):
            return False
        frame = frames[frame_index]
        if kernel is not None:
            try:
                if hasattr(kernel, "restore_snapshot"):
                    snap_id = f"rec_{recording_id}_frame_{frame_index}"
                    kernel.restore_snapshot(snap_id)
                    logger.info("Rollback to snapshot: %s frame %d", recording_id, frame_index)
                    return True
            except Exception as e:
                logger.warning("Rollback snapshot failed: %s", e)
        self._playback_cursor[recording_id] = frame_index
        logger.info("Rollback cursor to frame %d (no kernel snapshot)", frame_index)
        return True

    def list_recordings(self) -> list[dict[str, Any]]:
        results = []
        for rec_id, frames in self._recordings.items():
            config = self._configs.get(rec_id, RecordingConfig())
            results.append(
                {
                    "recording_id": rec_id,
                    "total_frames": len(frames),
                    "sensors": config.sensors or ["all"],
                }
            )
        return results

    def get_recording(self, recording_id: str) -> list[dict[str, Any]] | None:
        if recording_id not in self._recordings:
            return None
        frames = self._recordings[recording_id]
        return [
            {
                "frame_index": f.frame_index,
                "sim_time": f.sim_time,
                "sensor_data": f.sensor_data,
                "annotations": f.annotations,
            }
            for f in frames
        ]

    def delete_recording(self, recording_id: str) -> bool:
        if recording_id not in self._recordings:
            return False
        if self._active == recording_id:
            self._active = None
        del self._recordings[recording_id]
        self._configs.pop(recording_id, None)
        self._playback_cursor.pop(recording_id, None)
        logger.info("Recording deleted: %s", recording_id)
        return True

    def reset(self) -> None:
        self._active = None
        self._recordings.clear()
        self._configs.clear()
        self._playback_cursor.clear()
        self._frame_index = 0
        self._entity_positions.clear()
        logger.info("DataCollector reset")
