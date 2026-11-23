# Export configs/paths.yaml as shell variables.  Source this from any script:
#     . "$(dirname "$0")/../../solo2stack/env.sh"
# Keys become upper-case variables: data_root -> $DATA_ROOT.
_s2s_root() {
  local d; d=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); echo "$d"
}
S2S_ROOT=$(_s2s_root)
S2S_PATHS="$S2S_ROOT/configs/paths.yaml"
if [ ! -f "$S2S_PATHS" ]; then
  echo "configs/paths.yaml not found; copy configs/paths.example.yaml first" >&2
  return 1 2>/dev/null || exit 1
fi
while IFS= read -r _line; do
  case "$_line" in \#*|"") continue ;; esac
  _k=${_line%%:*}
  _v=${_line#*:}
  _v=${_v%%#*}
  _v=$(echo "$_v" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/")
  [ -z "$_k" ] && continue
  _K=$(echo "$_k" | tr '[:lower:]' '[:upper:]')
  export "$_K=$_v"
done < "$S2S_PATHS"
unset _line _k _v _K
