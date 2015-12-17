#!/bin/bash

ARCHIVE=cifar-10-binary.tar.gz
curl -O http://www.cs.toronto.edu/~kriz/$ARCHIVE

echo Unpacking archive...
tar xf $ARCHIVE

julia convert.jl

echo "data/train.hdf5" >> train.txt
echo "data/test.hdf5" >> test.txt
