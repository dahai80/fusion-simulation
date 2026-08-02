from __future__ import annotations

import dataclasses
import logging

logger = logging.getLogger(__name__)


def apply_component_update(comp, data: dict) -> None:
    if not dataclasses.is_dataclass(comp) or isinstance(comp, type):
        return
    for key, value in data.items():
        if hasattr(comp, key):
            try:
                setattr(comp, key, value)
            except (AttributeError, dataclasses.FrozenInstanceError) as e:
                logger.debug("Cannot set %s on %s: %s", key, type(comp).__name__, e)
