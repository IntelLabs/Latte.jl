#!/usr/bin/env bash

TARGET_DIR="$1"

if [[ -z "$TARGET_DIR" ]] ; then
    TARGET_DIR=$(pwd)
fi

for dset in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
    t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
do
    curl -o $TARGET_DIR/$dset -O http://yann.lecun.com/exdb/mnist/$dset
    STEM=$(basename "${dset}" .gz)
    gunzip -c $TARGET_DIR/$dset > $TARGET_DIR/$STEM
done

julia convert.jl $TARGET_DIR

echo "$TARGET_DIR/train.hdf5" > train.txt
echo "$TARGET_DIR/test.hdf5" > test.txt
