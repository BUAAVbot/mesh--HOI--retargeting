#!/usr/bin/env bash
set -euo pipefail

root=/data/yihang.bai/experiments/stage2_latest_gallery08_20261102
out="$root/deliverable"
mkdir -p "$out"
font=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf

ffmpeg -y \
  -i "$root/spider_strict/replay.mp4" \
  -i "$root/sbto_strict/replay.mp4" \
  -f lavfi -i "color=c=0x172033:s=640x480:r=50:d=5.78" \
  -filter_complex "\
[0:v]scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,drawtext=fontfile=${font}:text='SPIDER - strict FAIL':x=(w-text_w)/2:y=18:fontsize=25:fontcolor=white:box=1:boxcolor=0x00000099[v0];\
[1:v]scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,drawtext=fontfile=${font}:text='SBTO - strict FAIL (best)':x=(w-text_w)/2:y=18:fontsize=25:fontcolor=white:box=1:boxcolor=0x00000099[v1];\
[2:v]drawtext=fontfile=${font}:text='PRIME + Crocoddyl':x=(w-text_w)/2:y=120:fontsize=32:fontcolor=white,drawtext=fontfile=${font}:text='SOLVER FAILED':x=(w-text_w)/2:y=190:fontsize=36:fontcolor=0xff6b6b,drawtext=fontfile=${font}:text='LLT Hessian not SPD':x=(w-text_w)/2:y=255:fontsize=25:fontcolor=white,drawtext=fontfile=${font}:text='No accepted trajectory / replay':x=(w-text_w)/2:y=305:fontsize=22:fontcolor=white[v2];\
[v0][v1][v2]hstack=inputs=3[v]" \
  -map "[v]" -an -c:v libx264 -crf 20 -preset medium -pix_fmt yuv420p \
  "$out/stage2_latest_three_way_comparison.mp4"
