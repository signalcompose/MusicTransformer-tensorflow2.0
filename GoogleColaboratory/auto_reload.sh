#!/bin/bash

for i in `seq 0 12`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  open -a "Google Chrome" https://colab.research.google.com/github/signalcompose/MusicTransformer-tensorflow2.0/blob/master/GoogleCoraboratory/MusicTransformer.ipynb
  sleep 3600
done
