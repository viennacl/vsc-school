#!/bin/bash

g++ amg.cpp -o amg -I. -lOpenCL -DVIENNACL_WITH_OPENCL -DVIENNACL_WITH_OPENMP -fopenmp -O3
