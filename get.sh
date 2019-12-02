#!/usr/bin/env bash

declare -a files=("grabimage.py" "predict.py" "predict_test.py" "interactive.py" "heart.csv" "MNISTStarter.py" "notMNISTStarter.py" "notMNIST.npz")

for f in "${files[@]}"; do
  curl -OJ "http://pages.cpsc.ucalgary.ca/~hudsonj/CPSC501F19/$f"
done
