from team_red.config import CONFIG
from team_red.models.backend import BackendMode

BACKEND = None

if CONFIG.backend.mode == BackendMode.DIRECT:
    from .direct import Direct as Backend

    BACKEND = Backend()
elif CONFIG.backend.mode == BackendMode.REST:
    msg = "REST backend not implemented yet."
    raise NotImplementedError(msg)
else:
    msg = "Invalid backend configured."
    raise RuntimeError(msg)
