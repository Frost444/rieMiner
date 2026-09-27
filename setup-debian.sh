#!/usr/bin/env bash
# Modular setup: no NVIDIA driver installation and no mining side effects.
set -euo pipefail
mode=${1:-check}
root="$(cd "$(dirname "$0")" && pwd)"
case "$mode" in
  check) command -v python3 >/dev/null || { printf 'Python3 missing. Run bash setup-debian.sh runtime.\n' >&2; exit 1; }; exec python3 "$root/tools/check-system.py" ;;
  runtime|build|repair-runtime) ;;
  *) printf 'Usage: bash setup-debian.sh [check|runtime|build|repair-runtime]\n' >&2; exit 2 ;;
esac
if [ "$(id -u)" != 0 ]; then exec sudo bash "$0" "$mode"; fi
. /etc/os-release
if [ "$ID" != debian ] || [[ "${VERSION_ID:-}" != 12 && "${VERSION_ID:-}" != 13 ]]; then
  printf 'Automated setup supports Debian 12 and 13 only; see docs/BUILD.md.\n' >&2; exit 2
fi
apt-get update
ssl=libssl3
if [ "$VERSION_ID" = 13 ]; then ssl=libssl3t64; fi
packages=(python3 python3-rich libgmp10 libgmpxx4ldbl "$ssl" libstdc++6)
if [ "$mode" = build ]; then
  boost=libboost-json-dev
  if [ "$VERSION_ID" = 12 ]; then boost=libboost-json1.81-dev; fi
  packages+=(build-essential cmake ninja-build "$boost" libgmp-dev libssl-dev)
fi
extra=()
if [ "$mode" = repair-runtime ]; then extra+=(--reinstall); fi
apt-get install -y --no-install-recommends "${extra[@]}" "${packages[@]}"
printf '\nDependencies installed. NVIDIA drivers/toolkit were NOT changed.\n'
if [ "$mode" = build ]; then printf 'To rebuild, install CUDA 12.8 toolkit and set CUDACXX; see docs/BUILD.md.\n'; fi
