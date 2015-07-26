# convolution

First of all, in order to execute these files, you must have a NVIDIA CUDA and OpenCL compatible GPU.

To compile the CUDA version, it is necessary to set up the CUDA environment, which normally consists in downloading a file and double-click it. The environment will set up automatically. Otherwise, it is necessary to ensure that CUDA compiler tool "nvcc" is available from the command line.

To compile the OpenCL version, you must check that the right path too the OpenCL library has been included on the top of convolution_opencl.cpp. It is set to work in a OS X environment.

To execute the files, this requires to have previously installed ImagePack in order to read images. Once it is installed, you  must check that all of its tools (convert, decode, etc) are also available from the command line.
