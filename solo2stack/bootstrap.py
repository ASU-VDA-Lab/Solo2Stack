"""Put the repo root (and optionally model/) on sys.path.

Scripts live at several depths, so the root is located by looking for the
solo2stack package rather than by counting parent directories.
"""
import os
import sys


def repo_root(start=None):
    d = os.path.dirname(os.path.abspath(start or __file__))
    while True:
        if os.path.isdir(os.path.join(d, "solo2stack")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("solo2stack package not found above %s" % start)
        d = parent


def add_repo(start=None, *extra):
    """Insert the repo root, plus any repo-relative dirs, on sys.path."""
    root = repo_root(start)
    for p in [root] + [os.path.join(root, e) for e in extra]:
        if p not in sys.path:
            sys.path.insert(0, p)
    return root
