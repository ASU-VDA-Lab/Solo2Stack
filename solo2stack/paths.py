"""Repo-root discovery and the external-path table.

Code, configs and anything written inside the repository are addressed
relative to the repo root, which is found by walking up from this file.
Everything that lives OUTSIDE the repository -- the packed datasets, the raw
PACT outputs, checkpoints, and the big result trees -- is addressed through
configs/paths.yaml, which is per-machine and not committed.

Usage:

    from solo2stack.paths import P, REPO, repo_path
    root = P["data_root"]                   # external, from paths.yaml
    cfg  = repo_path("configs/pact/bc_4.config")   # inside the repo
"""
import os

from .miniyaml import load

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO, "configs", "paths.yaml")
EXAMPLE = os.path.join(REPO, "configs", "paths.example.yaml")

_MISSING = """configs/paths.yaml not found.

Copy the template and point it at your machine:

    cp configs/paths.example.yaml configs/paths.yaml
    $EDITOR configs/paths.yaml

It is intentionally not committed: every entry is a local absolute path."""


class _Paths(object):
    """Lazy, read-only view of configs/paths.yaml."""

    def __init__(self):
        self._d = None

    def _load(self):
        if self._d is None:
            if not os.path.exists(CONFIG):
                raise SystemExit(_MISSING)
            self._d = load(CONFIG)
        return self._d

    def __getitem__(self, key):
        d = self._load()
        if key not in d:
            raise SystemExit(
                "configs/paths.yaml is missing the key %r.\n"
                "configs/paths.example.yaml lists every key the code uses."
                % key)
        return d[key]

    def get(self, key, default=None):
        return self._load().get(key, default)

    def __contains__(self, key):
        return key in self._load()

    def keys(self):
        return self._load().keys()


P = _Paths()


def repo_path(*parts):
    """A path inside the repository, from the repo root."""
    return os.path.join(REPO, *parts)
