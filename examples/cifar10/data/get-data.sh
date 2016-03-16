#!/bin/bash
TARGET_DIR="$1"

if [[ -z "$TARGET_DIR" ]] ; then
    TARGET_DIR=$(pwd)
fi

ARCHIVE=cifar-10-binary.tar.gz
curl -o $TARGET_DIR/$ARCHIVE -O http://www.cs.toronto.edu/~kriz/$ARCHIVE

echo Unpacking archive...
tar xf $TARGET_DIR/$ARCHIVE -C $TARGET_DIR

julia convert.jl $TARGET_DIR

echo "$TARGET_DIR/train.hdf5" > train.txt
echo "$TARGET_DIR/test.hdf5" > test.txt
