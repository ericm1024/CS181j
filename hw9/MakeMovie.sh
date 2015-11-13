#!/bin/bash

echo 1>&2 "$0: Usage: \"sh MakeMovie.sh numberOfThreads numberOfIntervals"

if [ $# -lt 1 ]; then
  echo 1>&2 "$0: not enough arguments, need to provide the number of threads like \"sh MakeMovie.sh numberOfThreads\""
  exit 1
fi
numberOfIntervals=100000
if [ $# -gt 1 ]; then
  numberOfIntervals=$2
fi

#executable=ffmpeg
executable=avconv
numberOfThreadsString=$(printf '_%02d' $1)
numberOfIntervalsString=$(printf '_%06d' $numberOfIntervals)
${executable} -r 25 -f image2 -i figures/Threaded1DFDWave${numberOfIntervalsString}${numberOfThreadsString}_%05d.jpg -qscale 0 -vcodec mpeg4 Threaded1DFDWave${numberOfIntervalsString}$numberOfThreadsString.mp4
