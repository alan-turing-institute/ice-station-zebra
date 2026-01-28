from logging import getLogger

from ice_station_zebra.data_processors.filters import register_filters
from ice_station_zebra.data_processors.sources import register_sources
from ice_station_zebra.xpu import register_accelerators

logger = getLogger(__name__)


def register_plugins() -> None:
    """Register all plugins."""
    logger.info("Registering plugins...")
    register_accelerators()
    register_filters()
    register_sources()
