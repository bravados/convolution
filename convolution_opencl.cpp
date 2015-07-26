#include "CImg.h"
#include <iostream>
#include <fstream>


#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


using namespace cimg_library;


#define MAX_SOURCE_SIZE (0x100000)

int findClosestCeilingMultiple(int input_number, int reference_number){
    
    int input_number_aux = input_number;
    while(input_number_aux%reference_number != 0){
        input_number_aux++;
    }

    return input_number_aux;
}

/**

Calcula el máximo número de filas que podrán pasarse al device en base al temaño máximo libre

*/
int calculateMaxImageRows(cl_device_id device, int width, int channels, int kernel_size){

    int row_size = width+(kernel_size-1);

    cl_int memory;
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &memory, NULL);

    //std::cout << "El num. max. de filas es: " << ((memory/channels) - ((kernel_size-1)*row_size)) / (row_size+width) << std::endl;
    return ((memory/channels) - ((kernel_size-1)*row_size)) / (row_size+width);
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
void ImageConvolution(unsigned char * M, float * K, unsigned char * R, int width, int height, int colors, int kernel_size, int block_size){

    unsigned char * M_splitted;    //Piece of the image with the borders included
    int m_splitted_size;

    unsigned char * R_splitted; //Piece of the image result, without the extra borders
    int r_splitted_size;

    int max_rows;   //Max number of rows to execute according to the free device memory
    int from = 0, to;    //the rows to work over   
    int new_height; //the height of the portion of the image (to-from+1)
    int offset=0; //the offset for the current batch (number of rows left behind)

    cl_ulong time_start, time_end;
    cl_ulong time = 0, time_aux;

    cl_int clerr;

    
    //Step 0: Reading the kernel

        FILE *fp;
        char fileName[] = "./kernel.cl";
        char *source_str;
        size_t source_size;
         
        fp = fopen(fileName, "r");
        if (!fp) {
            fprintf(stderr, "Failed to load kernel.\n");
            exit(1);
        }

        source_str = (char*)malloc(MAX_SOURCE_SIZE);
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);



    //Step 1: Set up environment

        //Use the first platform
        cl_platform_id platform;
        clerr = clGetPlatformIDs(1, &platform, NULL);

        //Use the first device
        cl_device_id device;
        clerr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

        cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};

        //Create the context
        cl_context ctx = clCreateContext(cps, 1, &device, NULL, NULL, &clerr);

        //Create the command queue
        cl_command_queue queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &clerr);


    max_rows = calculateMaxImageRows(device, width, colors, kernel_size);
    to = height-1 - from > max_rows ? from+max_rows-1 : height-1;

    new_height = to-from+1;

    m_splitted_size = ((width+kernel_size-1)*new_height + (kernel_size-1)*(width+kernel_size-1)) * colors;
    r_splitted_size = new_height*width*colors;



    //Step 2: Declare Images (reserving memory on the device)

        //Create space for the source image on the device
        cl_mem bufferSourceImage = clCreateBuffer(ctx, CL_MEM_READ_ONLY, m_splitted_size*sizeof(unsigned char), NULL, &clerr);
        //std::cout << "Tras reservar memoria para source image: " << clerr << std::endl;

        //Create space for the output image on the device
        cl_mem bufferOutputImage = clCreateBuffer(ctx, CL_MEM_READ_ONLY, r_splitted_size*sizeof(unsigned char), NULL, &clerr);
        //std::cout << "Tras reservar memoria para output image: " << clerr << std::endl;

        //Create space for the filter on the device
        cl_mem bufferFilter = clCreateBuffer(ctx, CL_MEM_READ_ONLY, kernel_size*kernel_size*sizeof(float), NULL, &clerr);
        //std::cout << "Tras reservar memoria para el filter: " << clerr << std::endl;


    //Step 3: Write the Input Data

        //Copy the filter to the device
        clEnqueueWriteBuffer(queue, bufferFilter, CL_TRUE, 0, kernel_size*kernel_size*sizeof(float), K, 0, NULL, NULL);


    //Step 5: Creating the program

        //Create Kernel Program from the source
        cl_program program = clCreateProgramWithSource(ctx, 1, (const char **)&source_str, (const size_t *)&source_size, &clerr);
         
        //Build Kernel Program
        clerr = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

        char programInfo [200];

        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, 200, programInfo, NULL);
        //std::cout << "CL_PROGRAM_BUILD_STATUS: " << clerr << " " << programInfo << std::endl;    
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 200, programInfo, NULL);
        //std::cout << "CL_PROGRAM_BUILD_LOG: " << clerr << " " << programInfo << std::endl;

        
         
        //Create OpenCL Kernel
        cl_kernel kernel = clCreateKernel(program, "convolve", &clerr);
        //std::cout << "Intento de creación del kernel: " << clerr << std::endl;

        //Parameters Settings and Execution happen inside the loop


    do{
        
        to = height-1 - from > max_rows ? from+max_rows-1 : height-1;   //last row to convolve (inclusive)
        new_height = to-from+1; //Current number or rows to convolve



        //std::cout << "Llega hasta: " << to << std::endl;


        m_splitted_size = ((width+kernel_size-1)*new_height + (kernel_size-1)*(width+kernel_size-1)) * colors;  //size of the portion (incluiding the extra borders)
        r_splitted_size = new_height*width*colors;  //size of the result (no borders)


        //Allocate M_splitted and R_splitted on main memory
        M_splitted = new unsigned char [m_splitted_size];
        R_splitted = new unsigned char [r_splitted_size];


        prepareSubMatrix(M, M_splitted, from, to, width, height, colors, kernel_size);

        
        //Copy the source image to the device
        clerr = clEnqueueWriteBuffer(queue, bufferSourceImage, CL_TRUE, 0, m_splitted_size*sizeof(unsigned char), M_splitted, 0, NULL, NULL);
        //std::cout << "Transferencia de la imagen al device: " << clerr << std::endl;


        //Set OpenCL Kernel Parameters
        clerr = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferSourceImage);
        //std::cout << "Asignando argumento 0: " << clerr << std::endl;
        clerr = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferOutputImage);
        //std::cout << "Asignando argumento 1: " << clerr << std::endl;
        clerr = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferFilter);
        //std::cout << "Asignando argumento 2: " << clerr << std::endl;
        clerr = clSetKernelArg(kernel, 3, sizeof(int), &width);
        //std::cout << "Asignando argumento 3: " << clerr << std::endl;
        clerr = clSetKernelArg(kernel, 4, sizeof(int), &height);
        //std::cout << "Asignando argumento 4: " << clerr << std::endl;
        clerr = clSetKernelArg(kernel, 5, sizeof(int), &kernel_size);
        //std::cout << "Asignando argumento 5: " << clerr << std::endl;
        clerr = clSetKernelArg(kernel, 6, sizeof(int), &colors);
        //std::cout << "Asignando argumento 6: " << clerr << std::endl;

         
        //Execute OpenCL Kernel
        int NDRange_x = findClosestCeilingMultiple(width,block_size);
        int NDRange_y = findClosestCeilingMultiple(new_height,block_size);
        size_t globalws[2] = {NDRange_x,NDRange_y};
        size_t localws[2] = {block_size,block_size};

        cl_event event;
        clerr= clEnqueueNDRangeKernel(queue,kernel, 2, NULL, globalws, localws, 0, NULL, &event);
        //std::cout << "Kernel ejecutado: " << clerr << std::endl;

        //Time measuring
        clerr = clWaitForEvents(1, &event);
        //std::cout << "Esperando al evento: " << clerr << std::endl;
        clerr = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        //std::cout << "Midiendo start: " << clerr << std::endl;
        clerr = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        //std::cout << "Kernel end: " << clerr << std::endl;
        time_aux = (time_end - time_start);
        time += time_aux;
        clerr= clReleaseEvent(event);
        
        //Read the output image back to the host
        clerr = clEnqueueReadBuffer(queue, bufferOutputImage, CL_TRUE, 0, r_splitted_size*sizeof(unsigned char), R_splitted, 0, NULL, NULL);
        //std::cout << "Transferencia de la imagen al host: " << clerr << std::endl;


        //Adding these local results to the global matrix
        addSubImageToGlobalImage(R, R_splitted, offset, width, height, new_height, colors);

        //Free MM variables
        delete [] M_splitted;
        delete [] R_splitted;

        offset += new_height;   //more rows are left behind
        from = to+1;    //first row of the next batch


    } while(height-from > 0);


    //Free device variables
    clReleaseMemObject(bufferSourceImage);
    clReleaseMemObject(bufferOutputImage);
    clReleaseMemObject(bufferFilter);


    std::cout << (time)/1000000.0 << std::endl;

    
}


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
        std::cout << "The mask was not specified." << std::endl << "Run it as: ./program <image_file> <kernel wight> <kernel_size> <file with the kernel> <block size>" << std::endl;
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
    int block_size = atoi(argv[5]);
    ImageConvolution(matrix, kernel, result, width, height, 3, kernel_size, block_size);
        
    /*
    //Copy from matrix to image
    for(int i=0; i<width; i++)
        for(int j=0; j<height; j++)
            for(int k=0; k<=2; k++)
                output_image(i,j,k) = result[height*width*k + width*j + i];
                */
    
    //output_image.save("result.jpg");
        
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