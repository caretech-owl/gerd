import logging

from git import Remote, Repo

from gerd import __version__

_LOGGER = logging.getLogger(__name__)

repo = Repo()
if repo.is_dirty() or repo.untracked_files:
    msg = "Repository is dirty, please commit or stash your changes"
    raise ValueError(msg)
try:
    origin: Remote = repo.remote("origin")
    repo.create_tag(f"v{__version__}")
    origin.push(tags=True)
except ValueError as e:
    _LOGGER.error("Errro: %s", e)
