#!/bin/bash
# Download and extract SuiteSparse Matrix Market files
# Downloads in parallel, extracts sequentially, and places .mtx files
# directly under the project root's "matrices" directory.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MATRIX_DIR="$PROJECT_DIR/matrices"

mkdir -p "$MATRIX_DIR"
cd "$MATRIX_DIR"

# Phase 1: download all archives in parallel
# Each download writes to its own .tar.gz file, so no conflicts possible.
# Only the slow network I/O is parallelized.

download_all() {
  local MATRIX_LIST="$1"

  while IFS='|' read -r NAME URL; do
    [[ -z "$NAME" || -z "$URL" ]] && continue
    [[ -f "$NAME.mtx" || -f "$NAME.tar.gz" ]] && { echo "[SKIP] $NAME"; continue; }
    echo "[DOWNLOAD] $NAME ..."
    if curl -L -o "$NAME.tar.gz" "$URL" 2>/dev/null; then
      echo "[DL-OK] $NAME"
    else
      echo "[ERROR] download failed for $NAME"
    fi
  done <<< "$MATRIX_LIST"
}

# Phase 2: extract all archives sequentially
# This is fast (local disk I/O) so no parallelism needed, and avoids
# any race conditions when moving .mtx files around.

extract_all() {
  local MATRIX_LIST="$1"

  while IFS='|' read -r NAME URL; do
    [[ -z "$NAME" || -z "$URL" ]] && continue
    local TARFILE="$NAME.tar.gz"
    [[ ! -f "$TARFILE" ]] && continue

    echo "[EXTRACT] $NAME ..."
    if ! tar -xzf "$TARFILE"; then
      echo "[ERROR] extraction failed for $NAME"
      rm -f "$TARFILE"
      continue
    fi
    rm -f "$TARFILE"

    local MATRIX_FILE
    MATRIX_FILE=$(find "$NAME" -maxdepth 2 -name "*.mtx" 2>/dev/null | head -n1)
    if [[ -n "$MATRIX_FILE" && -f "$MATRIX_FILE" ]]; then
      mv "$MATRIX_FILE" ./
      rmdir "$NAME" 2>/dev/null || rm -rf "$NAME"
    fi
    echo "[OK] $NAME"
  done <<< "$MATRIX_LIST"
}

MATRIX_LIST="
FullChip|https://sparse.tamu.edu/MM/Freescale/FullChip.tar.gz
Ga41As41H72|https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Ga41As41H72.tar.gz
Si41Ge41H72|https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Si41Ge41H72.tar.gz
bone010|https://sparse.tamu.edu/MM/Oberwolfach/bone010.tar.gz
eu-2005|https://suitesparse-collection-website.herokuapp.com/MM/LAW/eu-2005.tar.gz
ldoor|https://sparse.tamu.edu/MM/GHS_psdef/ldoor.tar.gz
rajat31|https://sparse.tamu.edu/MM/Rajat/rajat31.tar.gz
ASIC_680ks|https://sparse.tamu.edu/MM/Sandia/ASIC_680ks.tar.gz
Rucci1|https://sparse.tamu.edu/MM/Rucci/Rucci1.tar.gz
boyd2|https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/boyd2.tar.gz
webbase-1M|https://sparse.tamu.edu/MM/Williams/webbase-1M.tar.gz
"

# Download phase: fork a subshell per matrix for true parallelism
while IFS='|' read -r NAME URL; do
  [[ -z "$NAME" || -z "$URL" ]] && continue
  [[ -f "$NAME.mtx" || -f "$NAME.tar.gz" ]] && { echo "[SKIP] $NAME"; continue; }
  echo "[DOWNLOAD] $NAME ..."
  {
    if curl -L -o "$NAME.tar.gz" "$URL" 2>/dev/null; then
      echo "[DL-OK] $NAME"
    else
      echo "[ERROR] download failed for $NAME"
    fi
  } &
done <<< "$MATRIX_LIST"

wait
echo ""
echo "Extracting..."

extract_one() {
  local NAME="$1"
  local TARFILE="$NAME.tar.gz"
  [[ ! -f "$TARFILE" ]] && return 0

  echo "[EXTRACT] $NAME ..."
  if ! tar -xzf "$TARFILE"; then
    echo "[ERROR] extraction failed for $NAME"
    rm -f "$TARFILE"
    return 1
  fi
  rm -f "$TARFILE"

  local MATRIX_FILE
  MATRIX_FILE=$(find "$NAME" -maxdepth 2 -name "*.mtx" 2>/dev/null | head -n1)
  if [[ -n "$MATRIX_FILE" && -f "$MATRIX_FILE" ]]; then
    mv "$MATRIX_FILE" ./
    rmdir "$NAME" 2>/dev/null || rm -rf "$NAME"
  fi
  echo "[OK] $NAME"
}

# Extract phase: sequential (fast, no races)
while IFS='|' read -r NAME URL; do
  [[ -z "$NAME" || -z "$URL" ]] && continue
  extract_one "$NAME"
done <<< "$MATRIX_LIST"

echo ""
echo "Done. .mtx files:"
ls -1 *.mtx 2>/dev/null
