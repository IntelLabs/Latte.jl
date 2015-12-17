#! /bin/bash
cd IO
cmake .
make
mv libLatteIO.so ../
cd ../communication
cmake .
make
mv libLatteComm.so ../
