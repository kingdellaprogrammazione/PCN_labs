#!/usr/bin/env bash
# Compress report.pdf by downsampling embedded images with Ghostscript.
# Usage: scripts/compress_pdf.sh [input.pdf] [output.pdf] [quality]
#   quality: screen (72dpi) | ebook (150dpi) | printer (300dpi, default) | prepress (300dpi, color-preserving)
set -euo pipefail

INPUT="${1:-report.pdf}"
OUTPUT="${2:-report_lowres.pdf}"
QUALITY="${3:-printer}"

case "$QUALITY" in
  screen|ebook|printer|prepress) ;;
  *)
    echo "Unknown quality '$QUALITY'. Use: screen | ebook | printer | prepress" >&2
    exit 1
    ;;
esac

if [ ! -f "$INPUT" ]; then
  echo "Input file not found: $INPUT" >&2
  exit 1
fi

gs -sDEVICE=pdfwrite \
   -dCompatibilityLevel=1.4 \
   -dPDFSETTINGS="/${QUALITY}" \
   -dNOPAUSE -dQUIET -dBATCH \
   -sOutputFile="$OUTPUT" \
   "$INPUT"

BEFORE=$(du -h "$INPUT" | cut -f1)
AFTER=$(du -h "$OUTPUT" | cut -f1)
echo "$INPUT ($BEFORE) -> $OUTPUT ($AFTER) [quality: $QUALITY]"
