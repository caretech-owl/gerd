from team_red.config import CONFIG
from team_red.models.backend import BackendMode

BACKEND = None

if CONFIG.backend.mode == BackendMode.DIRECT:
    from .direct import Direct as Backend

    BACKEND = Backend()
elif CONFIG.backend.mode == BackendMode.REST:
    raise NotImplementedError("REST backend not implemented yet.")
else:
    raise RuntimeError("Invalid backend configured.")
