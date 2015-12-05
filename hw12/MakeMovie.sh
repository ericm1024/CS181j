#!/bin/bash


if [ $# -lt 1 ]; then
  echo 1>&2 "$0: not enough arguments, need to provide the number of simulation processes"
  exit 1
fi

numberOfProcesses=$(printf '_%03d' $1)
echo $numberOfProcesses
avconv -r 25 -f image2 -i figures/Parallel1DFDWave${numberOfProcesses}_%05d.jpg -qscale 0 -vcodec mpeg4 Parallel1DFDWave$numberOfProcesses.mp4
