#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib
$HOME/caffe/build/tools/caffe time --model=./vgg_a.prototxt --iterations=10
