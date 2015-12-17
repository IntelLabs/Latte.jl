// Copyright (c) 2015 Intel Corporation. All rights reserved.
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mpi.h>
#include "hdf5.h"

#define MPI_OUT std::cout << "Worker " << MPI::COMM_WORLD.Get_rank() << ": "

hid_t create_hdf5_file(std::string file_name) {
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI::COMM_WORLD, MPI::INFO_NULL);

  std::string hdf5_file_name(file_name);
  MPI_OUT << "Creating HDF5 File:" << hdf5_file_name << std::endl;

  hid_t file_id = H5Fcreate(hdf5_file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);
  return file_id;
}

void read_metadata(std::string metadata_file,
                   std::vector<std::pair<std::string, int> > &lines) {
  MPI_OUT << "Parsing image list: " << metadata_file << std::endl;

  std::ifstream image_list(metadata_file.c_str());
  std::string image_file;
  int label;
  while (image_list >> image_file >> label) {
    lines.push_back(std::make_pair(image_file, label));
  }
}

int main(int argc, char *argv[]) {
  MPI::Init(argc, argv);
  int mpi_rank = MPI::COMM_WORLD.Get_rank();
  int mpi_size = MPI::COMM_WORLD.Get_size();

  if (argc < 3) {
    std::cout << "Error: Usage - convert $target_file_name $metadata_file $(mean_file_name, optional)" << std::endl;
    return -1;
  }
  bool compute_mean = argc == 4;
  std::string target_file_name(argv[1]);
  std::string metadata_file(argv[2]);

  hid_t file_id = create_hdf5_file(target_file_name);

  std::vector<std::pair<std::string, int> > lines;
  read_metadata(metadata_file, lines);
  int channels = 3;
  int height = 256;
  int width = 256;
  float *mean;
  if (compute_mean) {
      mean = new float[channels * height * width];
      std::fill(mean, mean+channels*height*width, 0);
  }

  hsize_t dim_data[] = {lines.size(), channels, height, width};
  hsize_t dim_label[] = {lines.size(), 1};
  hid_t data_dataspace = H5Screate_simple(4, dim_data, NULL);
  hid_t label_dataspace = H5Screate_simple(2, dim_label, NULL);

  MPI_OUT << "Creating Datasets" << std::endl;
  hid_t dset_data_id = H5Dcreate(file_id, "data", H5T_NATIVE_FLOAT,
      data_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  hid_t dset_label_id = H5Dcreate(file_id, "label", H5T_NATIVE_FLOAT,
      label_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  H5Sclose(data_dataspace);
  H5Sclose(label_dataspace);

  int chunk_size = lines.size() / mpi_size + 1;
  hsize_t start = mpi_rank * chunk_size;
  hsize_t end = (mpi_rank + 1) * chunk_size;

  hid_t data_slab_space = H5Dget_space(dset_data_id);
  hid_t label_slab_space = H5Dget_space(dset_label_id);
  hsize_t data_count[] = {1, channels, height, width};
  hsize_t label_count[] = {1, 1};
  hid_t data_memspace = H5Screate_simple(4, data_count, NULL);
  hid_t label_memspace = H5Screate_simple(2, label_count, NULL);

  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
  herr_t status;
  MPI_OUT << "Beginning conversion" << std::endl;
  float im_to_store[height*width*channels];
  for (hsize_t i = start; i < end; i++) {
    cv::Mat image, im_resized, float_im;
    hsize_t label_offset[] = {i, 0};
    hsize_t data_offset[] = {i, 0, 0, 0};
    float *label, *data;
    if (i < lines.size()) {
      image = cv::imread(lines[i].first, CV_LOAD_IMAGE_COLOR);
      image.convertTo(float_im, CV_32FC3);
      cv::resize(float_im, im_resized, cv::Size(height, width));
      H5Sselect_hyperslab(data_slab_space, H5S_SELECT_SET, data_offset, NULL,
          data_count, NULL);
      H5Sselect_hyperslab(label_slab_space, H5S_SELECT_SET, label_offset, NULL,
          label_count, NULL);
      float float_label = (float) lines[i].second;
      label = &float_label;
      for (int col=0; col < height; col++) {
        for (int row=0; row < width; row++) {
          for (int channel=0; channel < channels; channel++) {
            float val = (float) ((float*) im_resized.data)[col * width * channels + row * channels + channel];
            im_to_store[channel * height * width + row * width + col] = val;
            if (compute_mean) {
              mean[channel * height * width + row * width + col] += val;
            }
          }
        }
      }
      data = im_to_store;
    } else {
      H5Sselect_none(data_memspace);
      H5Sselect_none(data_slab_space);
      H5Sselect_none(label_memspace);
      H5Sselect_none(label_slab_space);
      data = NULL;
      label = NULL;
    }
    status = H5Dwrite(dset_data_id, H5T_NATIVE_FLOAT, data_memspace, data_slab_space, plist_id, data);
    status = H5Dwrite(dset_label_id, H5T_NATIVE_FLOAT, label_memspace, label_slab_space, plist_id, label);
    if (((int) i - start) % 100 == 0) {
      MPI_OUT << "Finished " << i - start << " of " << end - start << " images" << std::endl;
    }
  }
  MPI_OUT << "Completed conversion" <<  std::endl;

  MPI_OUT << "Cleaning up" <<  std::endl;
  H5Pclose(plist_id);
  H5Dclose(dset_data_id);
  H5Dclose(dset_label_id);
  H5Fclose(file_id);
  if (compute_mean) {
      MPI_OUT << "Computing Mean" <<  std::endl;
      float global_mean[channels * height * width];
      MPI_Reduce(mean, global_mean, channels*height*width, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      int rank;
      MPI_Comm_rank( MPI_COMM_WORLD, &rank );
      if (rank == 0) {
          for (int i=0; i < channels*height*width; i++) {
              global_mean[i] /= lines.size();
          }
      }
      MPI_OUT << "Writing Mean to File" <<  std::endl;
      hid_t mean_file = create_hdf5_file(std::string(argv[3]));
      hsize_t dim[] = {channels, height, width};
      hid_t dataspace = H5Screate_simple(3, dim, NULL);
      hid_t dset_id = H5Dcreate(mean_file, "mean", H5T_NATIVE_FLOAT, 
              dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (rank == 0) {
          MPI_OUT << "Finished writing mean" <<  std::endl;
          H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, global_mean);
      }
      H5Dclose(dset_id);
      H5Fclose(mean_file);
  }
  MPI_OUT << "Finalizing" <<  std::endl;
  MPI::Finalize();
  return 0;
}
