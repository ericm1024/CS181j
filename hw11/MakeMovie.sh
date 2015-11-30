#!/bin/bash

if [ $# -lt 1 ]; then
  echo 1>&2 "$0: not enough arguments, need to provide the flavor like \"sh MakeMovie.sh Invasive"
  exit 1
fi

avconv -r 25 -f image2 -i figures/Cuda1DFDWave$1_%05d.jpg -qscale 0 -vcodec mpeg4 Cuda1DFDWave$1.mp4
