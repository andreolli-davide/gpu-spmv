#!/bin/bash
# Download all 29 GPU SpMV benchmark matrices from SuiteSparse collection
# Downloads in parallel, extracts sequentially

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MATRIX_DIR="$PROJECT_DIR/matrices"

mkdir -p "$MATRIX_DIR"
cd "$MATRIX_DIR"

echo "=== SuiteSparse Matrix Downloader ==="
echo "Downloading 29 matrices for GPU SpMV benchmarks..."
echo ""

# All 29 matrices with correct URLs
MATRIX_LIST="
pdb1HYS|https://suitesparse-collection-website.herokuapp.com/MM/Williams/pdb1HYS.tar.gz
consph|https://suitesparse-collection-website.herokuapp.com/MM/Williams/consph.tar.gz
mac_econ_fwd500|https://suitesparse-collection-website.herokuapp.com/MM/Williams/mac_econ_fwd500.tar.gz
mc2depi|https://suitesparse-collection-website.herokuapp.com/MM/Williams/mc2depi.tar.gz
cop20k_A|https://suitesparse-collection-website.herokuapp.com/MM/Williams/cop20k_A.tar.gz
circuit5M|https://suitesparse-collection-website.herokuapp.com/MM/Freescale/circuit5M.tar.gz
webbase-2001|https://suitesparse-collection-website.herokuapp.com/MM/LAW/webbase-2001.tar.gz
lp_nug20|https://suitesparse-collection-website.herokuapp.com/MM/Qaplib/lp_nug20.tar.gz
ASIC_320k|https://suitesparse-collection-website.herokuapp.com/MM/Sandia/ASIC_320k.tar.gz
ASIC_680k|https://suitesparse-collection-website.herokuapp.com/MM/Sandia/ASIC_680k.tar.gz
webbase-1M|https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz
cnr-2000|https://suitesparse-collection-website.herokuapp.com/MM/LAW/cnr-2000.tar.gz
eu-2005|https://suitesparse-collection-website.herokuapp.com/MM/LAW/eu-2005.tar.gz
in-2004|https://suitesparse-collection-website.herokuapp.com/MM/LAW/in-2004.tar.gz
thermomech_dK|https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/thermomech_dK.tar.gz
ldoor|https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/ldoor.tar.gz
mip1|https://suitesparse-collection-website.herokuapp.com/MM/Andrianov/mip1.tar.gz
hvdc2|https://suitesparse-collection-website.herokuapp.com/MM/HVDC/hvdc2.tar.gz
fullb|https://suitesparse-collection-website.herokuapp.com/MM/DNVS/fullb.tar.gz
ins2|https://suitesparse-collection-website.herokuapp.com/MM/Andrianov/ins2.tar.gz
Ga41As41H72|https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Ga41As41H72.tar.gz
Si41Ge41H72|https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Si41Ge41H72.tar.gz
shipsec5|https://suitesparse-collection-website.herokuapp.com/MM/DNVS/shipsec5.tar.gz
windscreen|https://suitesparse-collection-website.herokuapp.com/MM/Oberwolfach/windscreen.tar.gz
thermal2|https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal2.tar.gz
offshore|https://suitesparse-collection-website.herokuapp.com/MM/Um/offshore.tar.gz
poisson3Db|https://suitesparse-collection-website.herokuapp.com/MM/FEMLAB/poisson3Db.tar.gz
Hook_1498|https://suitesparse-collection-website.herokuapp.com/MM/Janna/Hook_1498.tar.gz
nasasrb|https://suitesparse-collection-website.herokuapp.com/MM/Nasa/nasasrb.tar.gz
"

# Count total
TOTAL=$(echo "$MATRIX_LIST" | grep -c '|')
echo "Total matrices to download: $TOTAL"
echo ""

# Phase 1: Download all in parallel
echo "Phase 1: Downloading matrices..."
COUNT=0
while IFS='|' read -r NAME URL; do
  [[ -z "$NAME" || -z "$URL" ]] && continue
  COUNT=$((COUNT + 1))

  # Skip if already exists
  if [[ -f "$NAME.mtx" || -f "$NAME.tar.gz" ]]; then
    echo "  [$COUNT/$TOTAL] SKIP: $NAME (already exists)"
    continue
  fi

  echo "  [$COUNT/$TOTAL] Downloading: $NAME ..."
  {
    if curl -L -o "$NAME.tar.gz" "$URL" 2>/dev/null; then
      echo "    ✓ Downloaded: $NAME"
    else
      echo "    ✗ FAILED: $NAME"
    fi
  } &
done <<< "$MATRIX_LIST"

wait
echo ""
echo "Phase 1 complete. Waiting for all downloads to finish..."
echo ""

# Phase 2: Extract sequentially
echo "Phase 2: Extracting matrices..."
COUNT=0
while IFS='|' read -r NAME URL; do
  [[ -z "$NAME" || -z "$URL" ]] && continue
  COUNT=$((COUNT + 1))

  TARFILE="$NAME.tar.gz"
  [[ ! -f "$TARFILE" ]] && continue

  echo "  [$COUNT/$TOTAL] Extracting: $NAME ..."
  if tar -xzf "$TARFILE" 2>/dev/null; then
    # Find and move the .mtx file
    MATRIX_FILE=$(find "$NAME" -maxdepth 2 -name "*.mtx" 2>/dev/null | head -n1)
    if [[ -n "$MATRIX_FILE" && -f "$MATRIX_FILE" ]]; then
      mv "$MATRIX_FILE" ./
      rm -rf "$NAME" "$TARFILE"
      echo "    ✓ Extracted: $NAME"
    else
      echo "    ✗ No .mtx file found in $NAME"
      rm -rf "$NAME" "$TARFILE"
    fi
  else
    echo "    ✗ Extraction failed: $NAME"
    rm -f "$TARFILE"
  fi
done <<< "$MATRIX_LIST"

echo ""
echo "=== Download Complete ==="
echo ""
echo "Downloaded matrices (.mtx files):"
ls -1h *.mtx 2>/dev/null | nl
echo ""
MTXCOUNT=$(ls -1 *.mtx 2>/dev/null | wc -l)
echo "Total .mtx files: $MTXCOUNT / $TOTAL"
