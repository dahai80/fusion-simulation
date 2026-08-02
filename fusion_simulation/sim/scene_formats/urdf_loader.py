from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    from defusedxml import ElementTree
except ImportError:
    from xml.etree import ElementTree

logger = logging.getLogger(__name__)


class UrdfLoader:
    @staticmethod
    def parse(urdf_path: str) -> dict[str, Any]:
        p = Path(urdf_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"URDF file not found: {p}")
        tree = ElementTree.parse(p)
        root = tree.getroot()
        robot_name = root.attrib.get("name", p.stem)
        links = []
        joints = []
        for link_elem in root.findall("link"):
            link_name = link_elem.attrib.get("name", "")
            visual = link_elem.find("visual")
            collision = link_elem.find("collision")
            inertial = link_elem.find("inertial")
            links.append(
                {
                    "name": link_name,
                    "has_visual": visual is not None,
                    "has_collision": collision is not None,
                    "has_inertial": inertial is not None,
                }
            )
        for joint_elem in root.findall("joint"):
            joint_name = joint_elem.attrib.get("name", "")
            joint_type = joint_elem.attrib.get("type", "")
            parent = joint_elem.find("parent")
            child = joint_elem.find("child")
            axis = joint_elem.find("axis")
            limit = joint_elem.find("limit")
            joint_info: dict[str, Any] = {
                "name": joint_name,
                "type": joint_type,
                "parent_link": parent.attrib.get("link", "") if parent is not None else "",
                "child_link": child.attrib.get("link", "") if child is not None else "",
            }
            if axis is not None:
                joint_info["axis"] = [float(x) for x in (axis.text or "0 0 1").split()]
            if limit is not None:
                joint_info["lower"] = float(limit.attrib.get("lower", "0"))
                joint_info["upper"] = float(limit.attrib.get("upper", "0"))
                joint_info["effort"] = float(limit.attrib.get("effort", "0"))
                joint_info["velocity"] = float(limit.attrib.get("velocity", "0"))
            joints.append(joint_info)
        result = {
            "name": robot_name,
            "path": str(p),
            "links": links,
            "joints": joints,
            "num_links": len(links),
            "num_joints": len(joints),
        }
        logger.info("URDF parsed: %s (%d links, %d joints)", robot_name, len(links), len(joints))
        return result

    @staticmethod
    def get_joint_names(urdf_path: str) -> list[str]:
        info = UrdfLoader.parse(urdf_path)
        return [j["name"] for j in info["joints"]]

    @staticmethod
    def get_joint_limits(urdf_path: str) -> list[dict[str, float]]:
        info = UrdfLoader.parse(urdf_path)
        limits = []
        for j in info["joints"]:
            limits.append(
                {
                    "lower": j.get("lower", 0.0),
                    "upper": j.get("upper", 0.0),
                    "effort": j.get("effort", 0.0),
                    "velocity": j.get("velocity", 0.0),
                }
            )
        return limits
