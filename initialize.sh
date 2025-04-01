#!/usr/bin/bash 

mkdir -p models
mkdir -p data
mkdir -p benchmark

# download ConvMix benchmark
wget http://qa.mpi-inf.mpg.de/praise/benchmark.zip
unzip benchmark.zip 
rm benchmark.zip

# download data (including main results)
wget http://qa.mpi-inf.mpg.de/praise/data.zip
unzip data.zip 
rm data.zip

# download trained model checkpoints
wget http://qa.mpi-inf.mpg.de/praise/models.zip
unzip models.zip 
rm models.zip


echo "Successfully downloaded data!"