#! /bin/bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
echo "Unzipping tiny-imagenet, this will take awhile..."
unzip -q tiny-imagenet-200.zip
rm tiny-imagenet-200.zip
echo "Creating metadata files..."
julia create_metadata.jl
echo "Converting validation set"
../../../bin/convert_imagenet 64 ./val.hdf5 ./val_metadata.txt
echo "Converting train set"
../../../bin/convert_imagenet 64 ./train.hdf5 ./train_metadata.txt mean.hdf5
echo "data/train.hdf5" >> train.txt
echo "data/val.hdf5" >> val.txt
