#! /bin/bash
echo 'Building IO library.'
cd IO
mkdir -p build
rm -r build/*
cd build
cmake ..
make
mv libLatteIO.so ../../

if [ -z "$LATTE_BUILD_MPI" ]; then
    echo 'Skipping building communication library. If using Latte with MPI, set $LATTE_BUILD_MPI and rerun Pkg.build("Latte").'
else
    echo 'Building communication library.'
    cd ../../communication
    mkdir -p build
    rm -r build/*
    cd build
    cmake ..
    make
    mv libLatteComm.so .././
fi
