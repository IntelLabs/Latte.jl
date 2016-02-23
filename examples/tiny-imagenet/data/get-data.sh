# ! /bin/bash
TARGET_DIR="$1"
if [[ -z "$LATTE_ROOT" ]] ; then
    LATTE_ROOT="$HOME/.julia/v0.4/Latte"
fi

if [[ -z "$TARGET_DIR" ]] ; then
    TARGET_DIR="."
fi

# wget -P $TARGET_DIR http://cs231n.stanford.edu/tiny-imagenet-200.zip
# echo "Unzipping tiny-imagenet, this will take awhile..."
# unzip -q $TARGET_DIR/tiny-imagenet-200.zip -d $TARGET_DIR
rm $TARGET_DIR/tiny-imagenet-200.zip
echo "Creating metadata files..."
julia create_metadata.jl $TARGET_DIR
echo "Converting validation set"
$LATTE_ROOT/bin/convert_imagenet 64 $TARGET_DIR/val.hdf5 $TARGET_DIR/val_metadata.txt
echo "Converting train set"
$LATTE_ROOT/bin/convert_imagenet 64 $TARGET_DIR/train.hdf5 $TARGET_DIR/train_metadata.txt $TARGET_DIR/mean.hdf5
echo "$TARGET_DIR/train.hdf5" >> train.txt
echo "$TARGET_DIR/val.hdf5" >> val.txt
