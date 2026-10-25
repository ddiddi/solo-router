"""Solo Router: a semantic decision layer for robots and physical AI."""

from .config import RouterConfig, load_config
from .router import RouteResult, Router
from .signals import RobotRequest, Signals

__all__ = ["RobotRequest", "RouteResult", "Router", "RouterConfig", "Signals", "load_config"]
__version__ = "0.1.0"
