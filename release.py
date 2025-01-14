import logging
import subprocess

from git import Remote, Repo

from gerd import __version__

logging.basicConfig(level=logging.INFO)

_LOGGER = logging.getLogger("gerd.release")

repo = Repo()
if repo.is_dirty() or repo.untracked_files:
    msg = "Repository is dirty, please commit or stash your changes"
    raise ValueError(msg)
try:
    _LOGGER.info("Sync project...")
    subprocess.check_call(["uv", "sync"])  # noqa: S603, S607
    _LOGGER.info("Linting...")
    subprocess.check_call(["uv", "run", "poe", "lint"])  # noqa: S603, S607
    _LOGGER.info("Testing...")
    subprocess.check_call(["uv", "run", "poe", "test"])  # noqa: S603, S607
    origin: Remote = repo.remote("origin")
    tag = f"v{__version__}"
    _LOGGER.info("Create tag %s...", tag)
    repo.create_tag(tag)
    _LOGGER.info("Push tag %s to origin...", tag)
    origin.push(tags=True)
except ValueError as e:
    _LOGGER.error("Error: %s", e)
_LOGGER.info("Done!")
