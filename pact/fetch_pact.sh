#!/bin/bash
# Fetch PACT at the exact upstream commit this work was built on and apply the
# Solo2Stack patch.  PACT is third-party (GPL-3.0) and is therefore NOT vendored
# into this repository; only the patch is.
#
#   ./pact/fetch_pact.sh [target-dir]      default: ./pact/PACT
set -euo pipefail

UPSTREAM="https://github.com/peaclab/PACT.git"
COMMIT="bb7f05a0a893eac4826b1bc55b3e1f2415ba057d"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target="${1:-$here/PACT}"
patch_file="$here/solo2stack.patch"

if [ -e "$target" ]; then
  echo "target already exists: $target" >&2
  echo "remove it first, or pass a different directory" >&2
  exit 1
fi

echo "cloning $UPSTREAM"
git clone --quiet "$UPSTREAM" "$target"
git -C "$target" checkout --quiet "$COMMIT"
echo "checked out $COMMIT"

git -C "$target" apply --check "$patch_file"
git -C "$target" apply "$patch_file"
echo "applied $(basename "$patch_file")"

cat <<MSG

PACT is ready at:
    $target

Point configs/paths.yaml at it if you intend to run simulations, and use the
Python 3.6 environment described in requirements-pact.txt.  The solver is
SciPy's sparse LU; no external solver binary is needed.
MSG
