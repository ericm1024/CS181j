#!/bin/bash


if [ $# -lt 2 ]; then
  echo 1>&2 "$0: not enough arguments, need to provide the flavor and number of simulation processes like \"sh MakeMovie.sh NeedToKnow 4"
  exit 1
fi

numberOfProcesses=$(printf '_%03d' $2)
ffmpeg -r 25 -f image2 -i figures/Mini2dMD_$1${numberOfProcesses}_%05d.jpg -qscale 0 -vcodec mpeg4 Mini2dMD_$1$numberOfProcesses.mp4
