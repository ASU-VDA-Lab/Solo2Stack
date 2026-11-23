"""A deliberately tiny YAML reader for configs/paths.yaml.

The repository must run inside the two existing environments (Python 3.6 for
PACT, Python 3.10 for the model) without installing anything, and neither has
PyYAML.  configs/paths.yaml is therefore restricted to the subset parsed here:

    key: value          # scalars only, one per line
    # comment           # full-line comments
                        # blank lines

Values may be quoted with ' or ".  Anything else -- nesting, lists, anchors,
multi-line scalars -- raises, rather than being silently misread.  The file
stays valid YAML, so PyYAML would also read it correctly if it is available.
"""


class MiniYamlError(ValueError):
    pass


def load(path):
    """Parse a flat key: value file into a dict of strings."""
    out = {}
    with open(path) as fh:
        for n, raw in enumerate(fh, 1):
            line = raw.split("#", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
            if not line.strip():
                continue
            if line[0] in " \t":
                raise MiniYamlError(
                    "%s:%d: indented line; paths.yaml must be a flat "
                    "key: value mapping" % (path, n))
            if ":" not in line:
                raise MiniYamlError("%s:%d: no ':' in %r" % (path, n, raw.strip()))
            key, val = line.split(":", 1)
            key, val = key.strip(), val.strip()
            if not val:
                raise MiniYamlError(
                    "%s:%d: key %r has no value; nested blocks are not "
                    "supported" % (path, n, key))
            if val[0] in "[{&*|>":
                raise MiniYamlError(
                    "%s:%d: value %r uses YAML syntax this reader does not "
                    "support" % (path, n, val))
            if len(val) > 1 and val[0] == val[-1] and val[0] in "'\"":
                val = val[1:-1]
            out[key] = val
    return out
