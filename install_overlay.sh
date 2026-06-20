#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  ISAACLAB_ROOT="$1"
else
  ISAACLAB_ROOT="${ISAACLAB_ROOT:-}"
fi

if [[ -z "${ISAACLAB_ROOT}" ]]; then
  echo "Usage: ./install_overlay.sh /path/to/IsaacLab"
  echo "Or set ISAACLAB_ROOT=/path/to/IsaacLab"
  exit 1
fi

if [[ ! -f "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
  echo "Not an IsaacLab root: ${ISAACLAB_ROOT}"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${ISAACLAB_ROOT}/scripts"
mkdir -p "${ISAACLAB_ROOT}/source"
mkdir -p "${ISAACLAB_ROOT}/assets"

rsync -a "${REPO_ROOT}/scripts/" "${ISAACLAB_ROOT}/scripts/"
rsync -a "${REPO_ROOT}/source/" "${ISAACLAB_ROOT}/source/"
rsync -a "${REPO_ROOT}/assets/" "${ISAACLAB_ROOT}/assets/"

echo "Installed SO101 bimanual handover overlay into: ${ISAACLAB_ROOT}"
