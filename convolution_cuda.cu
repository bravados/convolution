#include "CImg.h"
#include <iostream>
#include <fstream>
#include <math.h>

using namespace cimg_library;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__constant__ float Kdc[100];

__global__
void convolve_global (const unsigned char * __restrict__ Md, unsigned char * Rd, int width, int height, int kernel_size, int tile_width, int channels){

    int row = blockIdx.y*tile_width + threadIdx.y;
    int col = blockIdx.x*tile_width + threadIdx.x;

    if(row < height  &&  col < width){

        int sum = 0;
        int pixel;
        int local_pixel;
        int working_pixel;

        int row_offset = (kernel_size/2)*(width+kernel_size-1);
        int col_offset = kernel_size/2;

        for(int color=0; color<channels; color++){

            pixel = color*width*height + row*width + col;
            local_pixel = color*(width+kernel_size-1)*(height+kernel_size-1) + row*(width+kernel_size-1) + col + row_offset + col_offset;
                for(int x=(-1)*kernel_size/2; x<=kernel_size/2; x++)
                    for(int y=(-1)*kernel_size/2; y<=kernel_size/2; y++){
                        working_pixel = local_pixel + x + y*(width+kernel_size-1);
                        sum += Md[working_pixel] * Kdc[x+kernel_size/2 + (y+kernel_size/2)*kernel_size];

                    }
                sum = sum < 0 ? 0 : sum > 255 ? 255 : sum;
                Rd[pixel] = (int) sum;
                sum = 0;
        }
    }
}

__global__
void convolve_shared (const unsigned char * __restrict__ Md, unsigned char * Rd, int width, int height, int kernel_size, int tile_width, int channels){

    extern __shared__ unsigned char Mds[];

    int row = blockIdx.y*tile_width + threadIdx.y;
    int col = blockIdx.x*tile_width + threadIdx.x;

    if(row < height  &&  col < width){

        int sum = 0;
        int pixel;  //the pixel to copy from Md (the input image)
        int local_pixel;    //the pixel in shared memory

        int start_pixel;    //the offset to copy the borders

        int mds_width = tile_width+kernel_size-1;
        int md_width = width+kernel_size-1;
        int md_height = height+kernel_size-1;

        for(int color=0; color<channels; color++){

            pixel = color*md_width*md_height + row*md_width + col  +  (kernel_size/2)*md_width + kernel_size/2; //position (including borders) + offset
            local_pixel = threadIdx.y*mds_width + threadIdx.x  +  (kernel_size/2)*mds_width + kernel_size/2;    //position + offset

  
            //Loading the pixels
            Mds[local_pixel] = Md[pixel];   //bringing the central pixel itself (position + offset)
  
            //Left edge of the block
            if(threadIdx.x == 0){
                start_pixel = mds_width*(kernel_size/2) + threadIdx.y*mds_width;
                for(int i=0; i<kernel_size/2; i++)
                    Mds[start_pixel + i] = Md[pixel-kernel_size/2 + i];
            }

            //Right edge of the block
            if(threadIdx.x == blockDim.x-1){
                start_pixel = mds_width*(kernel_size/2) + threadIdx.y*mds_width + kernel_size/2+tile_width;
                for(int i=0; i<kernel_size/2; i++)
                    Mds[start_pixel + i] = Md[pixel+1 + i];
            }


            //Top edge of the block
            if(threadIdx.y == 0){
                start_pixel = kernel_size/2 + threadIdx.x;
                for(int i=0; i<kernel_size/2; i++)
                    Mds[start_pixel + i*mds_width] = Md[pixel - (kernel_size/2)*md_width + i*md_width];
            }


            //Bottom edge of the block
            if(threadIdx.y == blockDim.y-1){
                start_pixel = ((kernel_size/2)+tile_width) * mds_width + kernel_size/2 + threadIdx.x;
                for(int i=0; i<kernel_size/2; i++)
                    Mds[start_pixel + i*mds_width] = Md[pixel+md_width + i*md_width];
            }

            //Top-Left edge
            if(threadIdx.x == 0  &&  threadIdx.y == 0){
                start_pixel = 0;
                for(int i=0; i<kernel_size/2; i++)
                    for(int j=0; j<kernel_size/2; j++)
                        Mds[i*mds_width + j] = Md[pixel - (kernel_size/2)*md_width - kernel_size/2 + i*md_width + j];
            }

            //Top-Right edge
            if(threadIdx.x == blockDim.x-1  &&  threadIdx.y == 0){
                start_pixel = kernel_size/2 + tile_width;
                for(int i=0; i<kernel_size/2; i++)
                    for(int j=0; j<kernel_size/2; j++)
                        Mds[start_pixel + i*mds_width + j] = Md[pixel+1 - (kernel_size/2)*md_width + i*md_width + j];
            }


            //Bottom-Left edge
            if(threadIdx.x == 0  &&  threadIdx.y == blockDim.y-1){
                start_pixel = ((kernel_size/2)+tile_width) * mds_width;
                for(int i=0; i<kernel_size/2; i++)
                    for(int j=0; j<kernel_size/2; j++)
                        Mds[start_pixel + i*mds_width + j] = Md[pixel + (kernel_size/2)*md_width - kernel_size/2 + i*md_width + j];
            }

            //Bottom-Right edge
            if(threadIdx.x == blockDim.x-1  &&  threadIdx.y == blockDim.y-1){
                start_pixel = ((kernel_size/2)+tile_width) * mds_width + kernel_size/2 + tile_width;
                for(int i=0; i<kernel_size/2; i++)
                    for(int j=0; j<kernel_size/2; j++)
                        Mds[start_pixel + i*mds_width + j] = Md[pixel + (kernel_size/2)*md_width + 1 + i*md_width + j];
            }

            __syncthreads();    //all threads collaborate to load the Mds matrix


            //Convolving
            for(int x=(-1)*kernel_size/2; x<=kernel_size/2; x++)
                for(int y=(-1)*kernel_size/2; y<=kernel_size/2; y++)
                    sum += Mds[local_pixel + x + y*mds_width] * Kdc[x+kernel_size/2 + (y+kernel_size/2)*kernel_size];
            sum = sum < 0 ? 0 : sum > 255 ? 255 : sum;
            Rd[color*width*height + row*width + col] = (int) sum;
            sum = 0;

            __syncthreads();    //to avoid pixels from Mds be overwritten without being processed
        }
    }

}


/**

Calcula el tamaño del bloque en base a:

-El tamaño de la imagen.
-Los slots de bloques físicos de la GPU.
-El número de registros físicos disponibles y los que se van usar en la función kernel.

Intenta maximizar el número de warps por bloque usando el menor número posible de slots.

*/
int calculateBlockSize(int width, int height, int slots_per_sm, int registers_per_thread){

    //Getting the properties of the device
    int device;
    cudaDeviceProp properties;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&properties, device);

    int max_threads_sm = properties.maxThreadsPerMultiProcessor;    //2048 iMac, 768 MacBook
    int max_block_size = properties.maxThreadsPerBlock; //1024 iMac, 512 MacBook
    int max_registers_sm = properties.regsPerBlock; //8192 MacBook


    //Calculating the size
    int max_concurrent_threads = max_threads_sm*registers_per_thread <= max_registers_sm ? max_threads_sm : max_registers_sm/registers_per_thread;    //that is the max number of threads I could ever execute at the same time (either because of the number of thread slots or the number of registers)
    float sm_threads = 0, sm_threads_aux;  //number of concurrent threads being executed in the same SM
    int slot_size;
    int slots;
    int block_size, block_size_aux;
    int total_threads;
    int total_blocks;
    float batches, batches_aux;  //number of times the SM will be loaded with threads

    //Maximize the number of warps
    for(int i=1; i<=slots_per_sm; i++){
        slot_size = max_concurrent_threads/i;
        if(slot_size <= max_block_size){
            block_size_aux = (int) sqrt(slot_size); //I'm looking for square kernels
            sm_threads_aux = i*block_size_aux*block_size_aux;
            
            total_blocks = (width/block_size_aux) * (height/block_size_aux);

            if(width%block_size_aux != 0)
                total_blocks += height/block_size_aux;
            if(height%block_size_aux != 0)
                total_blocks += width/block_size_aux;
            if(width%block_size_aux != 0  &&  height%block_size_aux != 0)
                total_blocks--;    //overlapping

            total_threads = total_blocks * block_size_aux*block_size_aux;
            batches_aux = (float)total_threads/sm_threads_aux;

            if(sm_threads_aux > sm_threads  ||  batches_aux < batches){ //always looking for the greater number of concurrent threads, avoiding the underutilization
                sm_threads = sm_threads_aux;
                batches = batches_aux;
                block_size = block_size_aux;
                slots = i;
                //std::cout << "Con " << block_size << "x" << block_size << " se usan " << sm_threads <<  " hebras por SM y hacen falta " << batches << " tandas. " << std::endl << "Se usan " << total_threads << " hebras con " << slots << " slots" << std::endl;
            }
        }
    }

    //std::cout << "El tamaño del bloque será: " << block_size << ", llenando: " << slots << " de " << slots_per_sm << " slots, usando: " << block_size*block_size*slots*registers_per_thread << " de " << max_registers_sm << " registros." << max_concurrent_threads-block_size*block_size*slots << " hebras quedan libres." << std::endl;

    return block_size;
}

/**

Calcula el máximo número de filas que podrán pasarse al device en base al temaño máximo libre

*/
int calculateMaxImageRows(int width, int channels, int kernel_size){

    int row_size = width+(kernel_size-1);

    //mem info
    size_t free, total;
    cudaMemGetInfo(&free, &total);

    free -= free/10;

    return ((free/channels) - ((kernel_size-1)*row_size)) / (row_size+width);
}

/**

Divide la imagen en una porción que quepa en el device. Devuelve el tamaño de la matriz preparada.

*/
void prepareSubMatrix(unsigned char *matrix, unsigned char *& subMatrix, int from_row, int to_row, int width, int height, int channels, int kernel_size){

    int rows = (to_row-from_row)+1;
    int row_size = width+kernel_size-1; //size of a row in subMatrix
    int layer_size_sub = row_size*rows + (kernel_size-1)*row_size;    //size of each color component in subMatrix

    int offset ;

    for(int c=0; c<channels; c++){

        offset = (kernel_size/2) * row_size;
        
        if(from_row == 0)
            //Filling the top border of black
            for(int k=0; k<kernel_size/2; k++)
                for(int l=0; l<row_size; l++)
                    subMatrix[c*layer_size_sub + k*row_size + l] = 0;
        else
            //Filling the top border of the pixels from the previous batch
            for(int k=0; k<kernel_size/2; k++)
                for(int l=0; l<row_size; l++)
                    subMatrix[c*layer_size_sub + k*row_size + l] = (l>=0  &&  l<kernel_size/2)  ||  l>= row_size-kernel_size/2   ?   0   :   matrix[(c*width*height) + from_row*width - (width*(kernel_size/2)) + (k*width) + l];



        //Copying the content putting the left and right borders as black
        for(int i=from_row, k=0; i<=to_row; i++, k++){

            for(int l=0; l<kernel_size/2; l++)
                subMatrix[c*layer_size_sub + offset + k*row_size + l] = 0;  //left border

            for(int j=0, l=kernel_size/2; j<width; j++, l++)
                subMatrix[c*layer_size_sub + offset + k*row_size + l] = matrix[c*width*height + i*width + j]; //center

            for(int l=row_size-kernel_size/2; l<row_size; l++)
                subMatrix[c*layer_size_sub + offset + k*row_size + l] = 0;  //right

        }
        
        offset = (kernel_size/2) * row_size  +  rows*row_size;


        if(to_row == height-1)
            //Filling the bottom border of black
            for(int k=0; k<kernel_size/2; k++)
                for(int l=0; l<row_size; l++)
                    subMatrix[c*layer_size_sub + offset + k*row_size + l] = 0;
        else
            //Filling the bottom border of the pixels from the previous batch
            for(int k=0; k<kernel_size/2; k++)
                for(int l=0; l<row_size; l++)
                    if((l>=0  &&  l<kernel_size/2)  ||  (l>= row_size-kernel_size/2  &&  l<row_size))
                        subMatrix[c*layer_size_sub + offset + k*row_size + l] = 0;
                    else
                        subMatrix[c*layer_size_sub + offset + k*row_size + l] = (l>=0  &&  l<kernel_size/2)  ||  l>= row_size-kernel_size/2   ?   0   :   matrix[(c*width*height) + (to_row + 1)*width + (k*width) + l];


    }
}

/*

Acopla la porción de imagen que se ha trabajado a la imagen global

*/
void addSubImageToGlobalImage(unsigned char * R, unsigned char * R_splitted, int row_offset, int width, int height, int new_height, int channels){
    for(int c=0; c<channels; c++)
        for(int i=0; i<new_height; i++)
            for(int j=0; j<width; j++)
                R[c*height*width + row_offset*width + i*width + j] = R_splitted[c*new_height*width + i*width + j];
}


/*

Aplica la convolución.

-Prepara las estructuras de datos.
-Divide la imagen en subconjuntos que quepan en el device.
-Envía la porción(es) de imagen al device.
-Llama al kernel.
-Recolecta los resultados de las porciones en un resultado global.

*/
void ImageConvolution(unsigned char * M, float * K, unsigned char * R, int width, int height, int colors, int kernel_size, int version){

    bool first_time = true;

    //Variables on the device
    unsigned char * Md; //M, the matrix, on the the device
    //float * Kd; //K, the kernel, on the device
    unsigned char * Rd; //R, the output matrix, on the device

    //Allocate and Transfer Kd on the device (GLOBAL MEMORY)
    //cudaMalloc((void**) &Kd, kernel_size*kernel_size*sizeof(float));
    //cudaMemcpy(Kd, K, kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice);

    //Transfer Kd on the device (CONSTANT MEMORY)
    cudaMemcpyToSymbol(Kdc, K, kernel_size*kernel_size*sizeof(float));

    
    unsigned char * M_splitted;    //Piece of the image with the borders included
    int m_splitted_size;

    unsigned char * R_splitted; //Piece of the image result, without the extra borders
    int r_splitted_size;

    int max_rows;   //Max number of rows to execute according to the free device memory
    int from = 0, to = height-1;    //the rows to work over   
    int new_height; //the height of the portion of the image (to-from+1)
    int offset=0; //the offset for the current batch (number of rows left behind)

    //Grid and Block dim
    int block_size;
    dim3 dimGrid;
    dim3 dimBlock;

    //Shared Memory
    int shared_mem_size;

    //Time measuring
    cudaEvent_t start, stop;
    float time = 0, time_aux;


    max_rows = calculateMaxImageRows(width, colors, kernel_size);
    to = height-1 - from > max_rows ? from+max_rows-1 : height-1;

    new_height = to-from+1;

    m_splitted_size = ((width+kernel_size-1)*new_height + (kernel_size-1)*(width+kernel_size-1)) * colors;
    r_splitted_size = new_height*width*colors;


    //Allocate Md and Rd on the device
    gpuErrchk(cudaMalloc((void**) &Md, m_splitted_size));
    gpuErrchk(cudaMalloc((void**) &Rd, r_splitted_size));



    do{
        
        to = height-1 - from > max_rows ? from+max_rows-1 : height-1;   //last row to convolve (inclusive)
        new_height = to-from+1; //Current number or rows to convolve

        m_splitted_size = ((width+kernel_size-1)*new_height + (kernel_size-1)*(width+kernel_size-1)) * colors;  //size of the portion (incluiding the extra borders)
        r_splitted_size = new_height*width*colors;  //size of the result (no borders)


        //Allocate M_splitted and R_splitted on main memory
        M_splitted = new unsigned char [m_splitted_size];
        R_splitted = new unsigned char [r_splitted_size];


        prepareSubMatrix(M, M_splitted, from, to, width, height, colors, kernel_size);


        //Transfer Md to the device memory
        gpuErrchk(cudaMemcpy(Md, M_splitted, m_splitted_size, cudaMemcpyHostToDevice));



        //Invocation to convolve
        block_size = calculateBlockSize(width,new_height,16,9);
        dimGrid.x = width/block_size + (width%block_size == 0 ? 0 : 1);   //an extra block will be needed if the division is not exact
        dimGrid.y = new_height/block_size + (new_height%block_size == 0 ? 0 : 1); //an extra block will be needed if the division is not exact
        dimBlock.x = block_size;
        dimBlock.y = block_size;

        //Calculating the shared memory size
        shared_mem_size = block_size*block_size + (block_size*(kernel_size/2)*4) + 4*(kernel_size/2)*(kernel_size/2); //center + each side of the block + the corners

        //Preparing time measuring
        gpuErrchk(cudaEventCreate(&start));
        gpuErrchk(cudaEventCreate(&stop));

        if(first_time){
            first_time = false;
            std::cout << dimGrid.x << "x" << dimGrid.y << " " << dimGrid.x*dimGrid.y*block_size*block_size << " " << block_size << "x" << block_size << " ";
        }
        
        //Calling the kernel
        gpuErrchk(cudaEventRecord(start,0));
        if(version == 0)
            convolve_global<<<dimGrid,dimBlock>>>(Md,Rd,width,new_height,kernel_size,block_size,colors);
        else
            convolve_shared<<<dimGrid,dimBlock,shared_mem_size>>>(Md,Rd,width,new_height,kernel_size,block_size,colors);
        gpuErrchk(cudaEventRecord(stop,0));
        gpuErrchk(cudaEventSynchronize(stop))
        gpuErrchk(cudaEventElapsedTime(&time_aux,start,stop));
        time += time_aux;

        //Transfer R from the device to host
        gpuErrchk(cudaMemcpy(R_splitted, Rd, r_splitted_size, cudaMemcpyDeviceToHost));


        //Adding these local results to the global matrix
        addSubImageToGlobalImage(R, R_splitted, offset, width, height, new_height, colors);

        //Free MM variables
        delete [] M_splitted;
        delete [] R_splitted;

        offset += new_height;   //more rows are left behind
        from = to+1;    //first row of the next batch


    } while(height-from > 0);


    //Free device variables
    cudaFree(Md);
    //cudaFree(Kd);
    cudaFree(Rd);


    std::cout << time <<std::endl;

    
}


/*

Prints the kernel

*/
void printKernel(float *kernel, int kernel_size){
    std::cout << "The kernel to be applied is:" << std::endl;
    for(int i=0; i<kernel_size*kernel_size; i++){
        std::cout << kernel[i];
        if((i+1)%kernel_size == 0)
            std::cout << std::endl;
        else
            std::cout << " ";
    }

    std::cout << std::endl;
}


int main(int argc, char* argv[]) {
    
    if(argc<6){
        std::cout << "The mask was not specified." << std::endl << "Run it as: ./program <image_file> <kernel wight> <kernel_size> <file with the kernel> <version {0|1}>" << std::endl;
        return 0;
    }


    //Getting the image
    CImg<unsigned char> image(argv[1]);
    int width = image.width();
    int height = image.height();
    int size = width*height*3;  //for every color component
        
        
    //Creating the output image
    CImg<unsigned char> output_image(width,height,1,3,0);   //z=1, dim=3 (colors), set to 0
    unsigned char * result = new unsigned char[size];
    
    
    //Getting the kernel from the file
    float weight = atoi(argv[2]);
    int kernel_size = atoi(argv[3]);
    float * kernel = new float [kernel_size*kernel_size];
    std::ifstream ifs(argv[4]);      
    for(int i=0; i<kernel_size*kernel_size  &&  !ifs.eof(); i++)
        ifs >> kernel[i];
    
    //Applying the weight
    for(int i=0; i<kernel_size*kernel_size; i++)
        kernel[i] = (float)kernel[i]/weight;

    //printKernel(kernel,kernel_size);
        
        
    //Creating the displays
    //CImgDisplay main_disp(image,"Image to convolve"), second_disp(output_image,"Image result");
    
        
        
    //Copy from image to a lineal matrix
    unsigned char * matrix = new unsigned char [size];
    unsigned char * data = image.data();
    
    for(int i=0; i<size; i++){
        matrix[i] = * data;
        data = data+sizeof(unsigned char);
    }


    //Convolving
    int version = atoi(argv[5]);
    ImageConvolution(matrix, kernel, result, width, height,3, kernel_size, version);
        
    
    //Copy from matrix to image
    for(int i=0; i<width; i++)
        for(int j=0; j<height; j++)
            for(int k=0; k<=2; k++)
                output_image(i,j,k) = result[height*width*k + width*j + i];
    
    output_image.save("result.jpg");
        /*
    
    second_disp.display(output_image);
    int x = main_disp.window_x() + main_disp.window_width();
    second_disp.move(x, 0);
    main_disp.wait();
    second_disp.wait();
    */
    
    delete [] matrix;
    delete [] result;   
    
    
    return 0;
}