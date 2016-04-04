#! /bin/bash
mkdir build
cd build
cmake ..
make
mv libLatteIO.so ../
mv libLatteComm.so ../
