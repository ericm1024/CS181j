#!/bin/bash


if [ $# -lt 2 ]; then
  echo 1>&2 "$0: not enough arguments, need to provide the number of simulation processes and the debug rank like \"sh MakeMovie.sh 4 1"
  exit 1
fi

numberOfProcesses=$(printf '_%03d_%03d' $1 $2)
echo $numberOfProcesses
ffmpeg -r 25 -f image2 -i figures/Mini2dMD_NeedToKnow_Debug${numberOfProcesses}_%05d.jpg -qscale 0 -vcodec mpeg4 Mini2dMD_NeedToKnow_Debug$numberOfProcesses.mp4
