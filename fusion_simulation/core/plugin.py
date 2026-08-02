from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    hooks: list[str] = field(default_factory=list)
    plugin_type: str = "generic"  # generic / sensor / physics_engine / task


_PLUGIN_DIRS: list[str] = []
_PLUGIN_REGISTRY: dict[str, type] = {}


class PluginSandbox:
    def __init__(
        self, allowed_modules: list[str] | None = None, max_memory_mb: float = 512.0, timeout_sec: float = 30.0
    ) -> None:
        self._allowed_modules = set(
            allowed_modules
            or [
                "numpy",
                "json",
                "math",
                "logging",
                "collections",
                "fusion_simulation.sensor",
                "fusion_simulation.physics",
                "fusion_simulation.core",
            ]
        )
        self._max_memory_mb = max_memory_mb
        self._timeout_sec = timeout_sec
        self._violations: list[str] = []

    def check_import(self, module_name: str) -> bool:
        root = module_name.split(".")[0]
        if root in self._allowed_modules:
            return True
        for allowed in self._allowed_modules:
            if module_name.startswith(allowed):
                return True
        self._violations.append(f"Blocked import: {module_name}")
        logger.warning("Sandbox blocked import: %s", module_name)
        return False

    def check_memory(self, current_mb: float) -> bool:
        if current_mb > self._max_memory_mb:
            self._violations.append(f"Memory limit: {current_mb:.1f}MB > {self._max_memory_mb:.1f}MB")
            logger.warning("Sandbox memory limit exceeded: %.1fMB > %.1fMB", current_mb, self._max_memory_mb)
            return False
        return True

    @property
    def violations(self) -> list[str]:
        return list(self._violations)

    def clear_violations(self) -> None:
        self._violations.clear()


class PluginManager:
    def __init__(self, sandbox: PluginSandbox | None = None) -> None:
        self._plugins: dict[str, Any] = {}
        self._hooks: dict[str, list[Callable]] = {}
        self._info: dict[str, PluginInfo] = {}
        self._sandbox = sandbox or PluginSandbox()
        self._sensor_factories: dict[str, Callable] = {}
        self._physics_factories: dict[str, Callable] = {}
        self._task_factories: dict[str, Callable] = {}
        logger.info("PluginManager created with sandbox")

    def register_plugin(self, name: str, plugin: Any, info: PluginInfo | None = None) -> None:
        if name in self._plugins:
            logger.warning("Plugin '%s' already registered, overwriting", name)
        self._plugins[name] = plugin
        if info is None:
            info = PluginInfo(name=name)
        self._info[name] = info
        if hasattr(plugin, "on_register"):
            try:
                plugin.on_register(self)
            except Exception as e:
                logger.error("Plugin '%s' on_register failed: %s", name, e)
        if info.plugin_type == "sensor" and hasattr(plugin, "create_sensor"):
            sensor_type = getattr(plugin, "sensor_type", name)
            self._sensor_factories[sensor_type] = plugin.create_sensor
            logger.info("Plugin '%s' registered sensor factory: %s", name, sensor_type)
        if info.plugin_type == "physics_engine" and hasattr(plugin, "create_engine"):
            engine_type = getattr(plugin, "engine_type", name)
            self._physics_factories[engine_type] = plugin.create_engine
            logger.info("Plugin '%s' registered physics factory: %s", name, engine_type)
        if info.plugin_type == "task" and hasattr(plugin, "create_task"):
            task_type = getattr(plugin, "task_type", name)
            self._task_factories[task_type] = plugin.create_task
            logger.info("Plugin '%s' registered task factory: %s", name, task_type)
        logger.info("Plugin registered: %s v%s type=%s", name, info.version, info.plugin_type)

    def unregister_plugin(self, name: str) -> bool:
        if name not in self._plugins:
            return False
        info = self._info.get(name)
        if info is not None:
            if info.plugin_type == "sensor":
                sensor_type = getattr(self._plugins[name], "sensor_type", name)
                self._sensor_factories.pop(sensor_type, None)
            if info.plugin_type == "physics_engine":
                engine_type = getattr(self._plugins[name], "engine_type", name)
                self._physics_factories.pop(engine_type, None)
            if info.plugin_type == "task":
                task_type = getattr(self._plugins[name], "task_type", name)
                self._task_factories.pop(task_type, None)
        plugin = self._plugins.pop(name)
        self._info.pop(name, None)
        if hasattr(plugin, "on_unregister"):
            try:
                plugin.on_unregister(self)
            except Exception as e:
                logger.error("Plugin '%s' on_unregister failed: %s", name, e)
        logger.info("Plugin unregistered: %s", name)
        return True

    def register_hook(self, hook_name: str, fn: Callable) -> None:
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(fn)
        logger.info("Hook registered: %s -> %s", hook_name, fn.__name__ if hasattr(fn, "__name__") else str(fn))

    def emit(self, hook_name: str, *args: Any, **kwargs: Any) -> list[Any]:
        results = []
        for fn in self._hooks.get(hook_name, []):
            try:
                result = fn(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error("Hook '%s' handler failed: %s", hook_name, e)
        return results

    def get_plugin(self, name: str) -> Any | None:
        return self._plugins.get(name)

    def list_plugins(self) -> list[dict[str, Any]]:
        return [
            {
                "name": info.name,
                "version": info.version,
                "description": info.description,
                "hooks": info.hooks,
                "plugin_type": info.plugin_type,
            }
            for info in self._info.values()
        ]

    def load_plugin_from_module(self, module_path: str, name: str = "") -> bool:
        try:
            root_module = module_path.split(".")[0]
            if not self._sandbox.check_import(module_path):
                logger.error("Sandbox blocked plugin load: %s", module_path)
                return False
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, "Plugin", None)
            if plugin_class is None:
                logger.error("Module '%s' has no Plugin class", module_path)
                return False
            plugin_name = name or getattr(module, "PLUGIN_NAME", module_path.split(".")[-1])
            plugin = plugin_class()
            info = PluginInfo(
                name=plugin_name,
                version=getattr(module, "PLUGIN_VERSION", "0.1.0"),
                description=getattr(module, "PLUGIN_DESCRIPTION", ""),
                plugin_type=getattr(module, "PLUGIN_TYPE", "generic"),
            )
            self.register_plugin(plugin_name, plugin, info)
            return True
        except Exception as e:
            logger.error("Failed to load plugin from '%s': %s", module_path, e)
            return False

    # --- Sensor integration ---

    def create_sensor(self, sensor_type: str, config: Any = None) -> Any | None:
        factory = self._sensor_factories.get(sensor_type)
        if factory is None:
            logger.warning("No sensor factory for type: %s", sensor_type)
            return None
        try:
            sensor = factory(config)
            logger.info("Plugin sensor created: %s", sensor_type)
            return sensor
        except Exception as e:
            logger.error("Plugin sensor creation failed: %s", e)
            return None

    def get_available_sensor_types(self) -> list[str]:
        return list(self._sensor_factories.keys())

    # --- Physics engine integration ---

    def create_physics_engine(self, engine_type: str, config: Any = None) -> Any | None:
        factory = self._physics_factories.get(engine_type)
        if factory is None:
            logger.warning("No physics factory for type: %s", engine_type)
            return None
        try:
            engine = factory(config)
            logger.info("Plugin physics engine created: %s", engine_type)
            return engine
        except Exception as e:
            logger.error("Plugin physics engine creation failed: %s", e)
            return None

    def get_available_engine_types(self) -> list[str]:
        return list(self._physics_factories.keys())

    # --- Task integration ---

    def create_task(self, task_type: str, config: Any = None) -> Any | None:
        factory = self._task_factories.get(task_type)
        if factory is None:
            logger.warning("No task factory for type: %s", task_type)
            return None
        try:
            task = factory(config)
            logger.info("Plugin task created: %s", task_type)
            return task
        except Exception as e:
            logger.error("Plugin task creation failed: %s", e)
            return None

    # --- Plugin discovery ---

    def discover_plugins(self, search_dirs: list[str] | None = None) -> list[dict[str, Any]]:
        dirs = search_dirs or _PLUGIN_DIRS
        discovered = []
        for search_dir in dirs:
            dir_path = Path(search_dir)
            if not dir_path.is_dir():
                continue
            for item in dir_path.iterdir():
                if item.is_dir() and (item / "__init__.py").exists():
                    plugin_info = {
                        "path": str(item),
                        "name": item.name,
                        "loadable": False,
                    }
                    init_file = item / "__init__.py"
                    try:
                        content = init_file.read_text(encoding="utf-8")
                        if "Plugin" in content or "PLUGIN_NAME" in content:
                            plugin_info["loadable"] = True
                    except Exception:
                        pass
                    discovered.append(plugin_info)
                elif item.is_file() and item.suffix == ".py":
                    plugin_info = {
                        "path": str(item),
                        "name": item.stem,
                        "loadable": False,
                    }
                    try:
                        content = item.read_text(encoding="utf-8")
                        if "Plugin" in content or "PLUGIN_NAME" in content:
                            plugin_info["loadable"] = True
                    except Exception:
                        pass
                    discovered.append(plugin_info)
        logger.info("Plugin discovery: %d found in %s", len(discovered), dirs)
        return discovered

    def load_discovered_plugins(self, search_dirs: list[str] | None = None) -> int:
        discovered = self.discover_plugins(search_dirs)
        loaded = 0
        for p in discovered:
            if not p.get("loadable", False):
                continue
            try:
                module_path = Path(p["path"])
                if module_path.is_dir():
                    parts = list(module_path.relative_to(Path.cwd()).parts)
                    mod_path = ".".join(parts)
                else:
                    parts = list(module_path.relative_to(Path.cwd()).with_suffix("").parts)
                    mod_path = ".".join(parts)
                if self.load_plugin_from_module(mod_path, p["name"]):
                    loaded += 1
            except Exception as e:
                logger.warning("Failed to load discovered plugin %s: %s", p["name"], e)
        logger.info("Loaded %d/%d discovered plugins", loaded, len(discovered))
        return loaded

    @property
    def sandbox(self) -> PluginSandbox:
        return self._sandbox

    def reset(self) -> None:
        for name in list(self._plugins.keys()):
            self.unregister_plugin(name)
        self._hooks.clear()
        self._sensor_factories.clear()
        self._physics_factories.clear()
        self._task_factories.clear()
        self._sandbox.clear_violations()
        logger.info("PluginManager reset")


def add_plugin_dir(directory: str) -> None:
    if directory not in _PLUGIN_DIRS:
        _PLUGIN_DIRS.append(directory)
        logger.info("Plugin directory added: %s", directory)


def get_plugin_dirs() -> list[str]:
    return list(_PLUGIN_DIRS)
