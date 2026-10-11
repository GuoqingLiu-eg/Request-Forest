import logging
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import (
    HookRegistry,
    HookSource,
    _current_plugin_source,
)

logger = logging.getLogger(__name__)


PLATFORM_PLUGINS_GROUP = "sglang.srt.platforms"
GENERAL_PLUGINS_GROUP = "sglang.srt.plugins"


_plugins_loaded = False


def load_plugins_by_group(
    group: str,
    excluded_dists: set[str] | None = None,
) -> dict[str, tuple[Callable[[], Any], str | None]]:


    allowed_set: set[str] | None = None
    allowed_str = envs.SGLANG_PLUGINS.get()
    if allowed_str:
        allowed_set = {x.strip() for x in allowed_str.split(",") if x.strip()}

    discovered = entry_points(group=group)
    if len(discovered) == 0:
        logger.debug("No plugins found for group %s.", group)
        return {}

    logger.info("Available plugins for group %s:", group)
    for ep in discovered:
        logger.info("  - %s -> %s", ep.name, ep.value)

    plugins: dict[str, tuple[Callable[[], Any], str | None]] = {}
    for ep in discovered:
        if allowed_set is not None and ep.name not in allowed_set:
            logger.info("Skipping plugin %s (not in SGLANG_PLUGINS)", ep.name)
            continue
        dist_name = ep.dist.name if ep.dist else None
        if excluded_dists and dist_name in excluded_dists:
            logger.info(
                "Skipping plugin %s (dist %s excluded by SGLANG_PLATFORM)",
                ep.name,
                dist_name,
            )
            continue
        try:
            func = ep.load()
            plugins[ep.name] = (func, dist_name)
            logger.info("Loaded plugin %s from group %s", ep.name, group)
        except Exception:
            logger.exception("Failed to load plugin %s from group %s", ep.name, group)

    return plugins


def _get_excluded_dists() -> set[str]:

    selected = envs.SGLANG_PLATFORM.get()
    if not selected:
        return set()
    platform_eps = entry_points(group=PLATFORM_PLUGINS_GROUP)
    return {ep.dist.name for ep in platform_eps if ep.dist and ep.name != selected}


def load_plugins():

    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True

    plugins = load_plugins_by_group(
        GENERAL_PLUGINS_GROUP,
        excluded_dists=_get_excluded_dists(),
    )

    for name, (func, dist_name) in plugins.items():
        source = HookSource(plugin_name=name, dist_name=dist_name)
        token = _current_plugin_source.set(source)
        try:
            func()
            logger.info("Executed general plugin: %s", name)
        except Exception:
            logger.exception("Failed to execute general plugin: %s", name)
        finally:
            _current_plugin_source.reset(token)


    HookRegistry.apply_hooks()
