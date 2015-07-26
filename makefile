CC=g++
CUDA=nvcc
CFLAGS=-w -I/opt/X11/include/
LDFLAGS=-L/opt/X11/lib/ -lX11
CUDA_NVCC_FLAGS= -arch=sm_30 -Xptxas="-v"
SRCS_SEQ= convolution_seq.cpp
SRCS_CUDA=convolution_cuda.cu
SRCS_OPENCL=convolution_opencl.cpp

OBJS_SEQ=convolution_seq.o
OBJS_CUDA=convolution_cuda.o
OBJS_OPENCL=convolution_opencl.o

EXEC_SEQ=convolution_seq
EXEC_CUDA=convolution_cuda
EXEC_OPENCL=convolution_opencl

all: $(EXEC_SEQ) $(EXEC_CUDA) $(EXEC_OPENCL)

	
$(EXEC_SEQ): $(OBJS_SEQ) 
	$(CC) -o $(EXEC_SEQ) $(OBJS_SEQ) $(LDFLAGS)

$(OBJS_SEQ): $(SRCS_SEQ)
	$(CC) -c $(SRCS_SEQ) $(CFLAGS)


$(EXEC_CUDA): $(SRCS_CUDA) 
	$(CUDA) -o $(EXEC_CUDA) $(SRCS_CUDA) $(CFLAGS) $(CUDA_NVCC_FLAGS) $(LDFLAGS)



$(EXEC_OPENCL): $(OBJS_OPENCL)
	$(CC) -o $(EXEC_OPENCL) $(OBJS_OPENCL) $(LDFLAGS) -framework OpenCL

$(OBJS_OPENCL): $(SRCS_OPENCL)
	$(CC) -c $(SRCS_OPENCL) $(CFLAGS)



clean:
	rm -rf *.o $(EXEC_SEQ) $(EXEC_CUDA) $(EXEC_OPENCL)
