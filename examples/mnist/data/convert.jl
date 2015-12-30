#=
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=#

# convert binary into HDF5 data
using HDF5

srand(12345678)

datasets = Dict("train" => ["train-labels-idx1-ubyte","train-images-idx3-ubyte"],
                "test" => ["t10k-labels-idx1-ubyte","t10k-images-idx3-ubyte"])

for key in keys(datasets)
  label_fn, data_fn = datasets[key]
  label_f = open(label_fn)
  data_f  = open(data_fn)

  label_header = read(label_f, Int32, 2)
  @assert ntoh(label_header[1]) == 2049
  n_label = round(Int, ntoh(label_header[2]))
  data_header = read(data_f, Int32, 4)
  @assert ntoh(data_header[1]) == 2051
  n_data = round(Int, ntoh(data_header[2]))
  @assert n_label == n_data
  h = round(Int, ntoh(data_header[3]))
  w = round(Int, ntoh(data_header[4]))

  n_batch = 1
  @assert n_data % n_batch == 0
  batch_size = round(Int, n_data / n_batch)

  println("Exporting $n_data digits of size $h x $w")

  h5open("$key.hdf5", "w") do h5
    dset_data = d_create(h5, "data", datatype(Float32), dataspace(w, h, 1, n_data))
    dset_label = d_create(h5, "label", datatype(Float32), dataspace(1, n_data))

    for i = 1:n_batch
      idx = (i-1)*batch_size+1:i*batch_size
      println("  $idx...")

      idx = collect(idx)
      rp = randperm(length(idx))

      img = readbytes(data_f, batch_size * h*w)
      img = convert(Array{Float32},img) / 256 # scale into [0,1)
      class = readbytes(label_f, batch_size)
      class = convert(Array{Float32},class)

      for j = 1:length(idx)
        r_idx = rp[j]
        dset_data[:,:,1,idx[j]] = img[(r_idx-1)*h*w+1:r_idx*h*w]
        dset_label[1,idx[j]] = class[r_idx]
      end
    end
  end

  close(label_f)
  close(data_f)
end

