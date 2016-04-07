# MNIST

## TLDR

```shell
$ cd ~/.julia/v0.4/Latte/examples/mnist/data
$ ./get-data.sh
$ cd ..
$ julia mnist.jl
```

## Preparing the data
These steps can be performed automatically by running the `get-data.sh` script
inside the `examples/mnist` directory.

First we will download the MNIST dataset from Yann Lecun's website.

```shell
TARGET_DIR=$(pwd)
for dset in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
    t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
do
    curl -o $TARGET_DIR/$dset -O http://yann.lecun.com/exdb/mnist/$dset
    STEM=$(basename "${dset}" .gz)
    gunzip -c $TARGET_DIR/$dset > $TARGET_DIR/$STEM
done
```

Next we will convert the binary data into HDF5 datasets.  This code
is contained in `convert.jl`.

This begins by reading the binary files using Julia's `open`.
```
base_dir = "./"

datasets = Dict("train" => ["$base_dir/train-labels-idx1-ubyte","$base_dir/train-images-idx3-ubyte"],
                "test" => ["$base_dir/t10k-labels-idx1-ubyte","$base_dir/t10k-images-idx3-ubyte"])

for key in keys(datasets)
  label_fn, data_fn = datasets[key]
  label_f = open(label_fn)
  data_f  = open(data_fn)
```

Next we read the headers for the binary data to give us information about the dataset dimensions.
```
  label_header = read(label_f, Int32, 2)
  @assert ntoh(label_header[1]) == 2049
  n_label = round(Int, ntoh(label_header[2]))
  data_header = read(data_f, Int32, 4)
  @assert ntoh(data_header[1]) == 2051
  n_data = round(Int, ntoh(data_header[2]))
  @assert n_label == n_data
  h = round(Int, ntoh(data_header[3]))
  w = round(Int, ntoh(data_header[4]))
```

Next we open an HDF5 file for writing and initialize two datasets (label and data).
```
  println("Exporting $n_data digits of size $h x $w")

  h5open("$base_dir/$key.hdf5", "w") do h5
    dset_data = d_create(h5, "data", datatype(Float32), dataspace(w, h, 1, n_data))
    dset_label = d_create(h5, "label", datatype(Float32), dataspace(1, n_data))
```

Then we read the label and data bytes and convert them to Arrays of Float32.  We normalize
the data to values between [0, 1) by dividing by 256.
```
    img = readbytes(data_f, n_data * h*w)
    img = convert(Array{Float32},img) / 256 # scale into [0,1)
    class = readbytes(label_f, n_data)
    class = convert(Array{Float32},class)
```

We will permute the indices of the dataset so that they are stored in a
shuffled ordering.  Then we iterate over the permuted indices and store
the data and label values into the HDF5 dataset.
```
    idx = 1:n_data
    println("  $idx...")

    idx = collect(idx)
    rp = randperm(length(idx))

    for j = 1:length(idx)
      r_idx = rp[j]
      dset_data[:,:,1,idx[j]] = img[(r_idx-1)*h*w+1:r_idx*h*w]
      dset_label[1,idx[j]] = class[r_idx]
    end
```

## The Model
The model code can be found in `examples/mnist.jl`
```
using Latte

net = Net(100)
data, label = HDF5DataLayer(net, "data/train.txt", "data/test.txt")
conv1    = ConvolutionLayer(:conv1, net, data, 20, 5, 1, 1)
relu1    = ReLULayer(:relu1, net, conv1)
pool1    = MaxPoolingLayer(:pool1, net, relu1, 2, 2, 0)
conv2    = ConvolutionLayer(:conv2, net, pool1, 50, 5, 1, 1)
relu2    = ReLULayer(:relu2, net, conv2)
pool2    = MaxPoolingLayer(:pool2, net, relu2, 2, 2, 0)
conv3    = ConvolutionLayer(:conv3, net, pool1, 50, 3, 1, 1)
relu3    = ReLULayer(:relu3, net, conv3)
pool3    = MaxPoolingLayer(:pool3, net, relu3, 2, 2, 0)
fc4      = InnerProductLayer(:fc4, net, pool3, 512)
relu4    = ReLULayer(:relu4, net, fc4)
fc5      = InnerProductLayer(:fc5, net, relu4, 512)
relu5    = ReLULayer(:relu5, net, fc5)
fc6      = InnerProductLayer(:fc6, net, relu5, 10)
loss     = SoftmaxLossLayer(:loss, net, fc6, label)
accuracy = AccuracyLayer(:accuracy, net, fc6, label)

params = SolverParameters(
    lr_policy    = LRPolicy.Inv(0.01, 0.0001, 0.75),
    mom_policy   = MomPolicy.Fixed(0.9),
    max_epoch    = 50,
    regu_coef    = .0005)
sgd = SGD(params)
solve(sgd, net)
```
