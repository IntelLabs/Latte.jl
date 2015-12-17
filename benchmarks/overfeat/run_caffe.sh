#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib
$HOME/caffe/build/tools/caffe time --model=./overfeat.prototxt --iterations=10
