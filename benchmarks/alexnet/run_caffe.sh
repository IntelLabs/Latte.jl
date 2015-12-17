#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib
$HOME/caffe/build/tools/caffe time --model=./alexnet.prototxt --iterations=10
