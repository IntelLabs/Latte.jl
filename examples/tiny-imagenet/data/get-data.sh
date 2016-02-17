# ! /bin/bash
TARGET_DIR="$1"
if [[ -z "$LATTE_ROOT" ]] ; then
    echo "Please set the LATTE_ROOT environment variable to your LATTE installation (i.e. export LATTE_ROOT=~/.julia/v0.4/Latte)"
    exit 1
fi
if [[ -z "TARGET_DIR" ]] ; then
    TARGET_DIR="./"
fi
cd $TARGET_DIR
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
echo "Unzipping tiny-imagenet, this will take awhile..."
unzip -q tiny-imagenet-200.zip
rm tiny-imagenet-200.zip
echo "Creating metadata files..."
julia create_metadata.jl
echo "Converting validation set"
$LATTE_ROOT/bin/convert_imagenet 64 ./val.hdf5 ./val_metadata.txt
echo "Converting train set"
$LATTE_ROOT/bin/convert_imagenet 64 ./train.hdf5 ./train_metadata.txt mean.hdf5
echo "data/train.hdf5" >> train.txt
echo "data/val.hdf5" >> val.txt
