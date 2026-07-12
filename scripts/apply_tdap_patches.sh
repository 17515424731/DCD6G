#!/usr/bin/env bash
set -euo pipefail

TDAP_ROOT="${TDAP_ROOT:-third_party/TDAP}"

if [ ! -d "${TDAP_ROOT}/.git" ]; then
  echo "TDAP repository not found at ${TDAP_ROOT}. Run scripts/fetch_third_party.sh first." >&2
  exit 1
fi

apply_patch_if_needed() {
  local patch_file="$1"
  local abs_patch
  abs_patch="$(cd "$(dirname "${patch_file}")" && pwd)/$(basename "${patch_file}")"
  if git -C "${TDAP_ROOT}" apply --check "${abs_patch}" >/dev/null 2>&1; then
    git -C "${TDAP_ROOT}" apply "${abs_patch}"
    echo "Applied ${abs_patch}"
  else
    echo "Skipped ${abs_patch}; it may already be applied or the upstream file changed."
  fi
}

PATCH_DIR="${PATCH_DIR:-patches}"
for patch_file in "${PATCH_DIR}"/*.patch; do
  [ -e "${patch_file}" ] || {
    echo "No patch files found in ${PATCH_DIR}" >&2
    exit 1
  }
  apply_patch_if_needed "${patch_file}"
done
