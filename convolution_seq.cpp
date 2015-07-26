#include "CImg.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>

using namespace cimg_library;


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
	
    if(argc<5){
        std::cout << "The mask was not specified." << std::endl << "Run it as: ./program <image_file> <kernel_size> <file with the kernel>" << std::endl;
        return 0;
    }
        
        
    //Getting the image
	CImg<unsigned char> image(argv[1]);
    int width = image.width();
    int height = image.height();
    int size = width*height*3;  //for every color component
        
        
    //Creating the output image
    CImg<unsigned char> output_image(width,height,1,3,0);   //z=1, dim=3 (colors), setted to 0
    
    
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

    printKernel(kernel,kernel_size);
    
    
    //Creating the displays
    CImgDisplay main_disp(image,"Image to convolve"), second_disp(output_image,"Image result");

    
    
    //Copy from image to buffer
    unsigned char *buffer = new unsigned char [size];
    unsigned char *data = image.data();
    
    for(int i=0; i<size; i++){
        buffer[i] = *data;
        data = data+sizeof(unsigned char);
    }
    data = image.data();
    
    
    //Preparing Measuring
    clock_t start, end; 
    start = clock(); 

    //Convolving
    int sum = 0;
    int pixel;
    int working_pixel;
    int row_start;
    int col_start;
    int kernel_start = (-1)*kernel_size/2;
    int kernel_end = kernel_size/2;
    for(int c=0; c<3; c++)  //RGB
        for(int i=0; i<height; i++) //rows
            for(int j=0; j<width; j++){  //coloumns
                pixel = c*width*height + i*width + j;
                for(int x=kernel_start; x<=kernel_end; x++)
                    for(int y=kernel_start; y<=kernel_end; y++){
                        working_pixel = pixel + x + y*width;
                        row_start = (pixel/width)*width + y*width;
                        col_start = pixel/(width*height)*width*height;
                        if(working_pixel >= row_start  &&  working_pixel < row_start+width  &&  working_pixel >= col_start  &&  working_pixel < col_start+width*height)
                            sum += *(data+working_pixel) * kernel[x+kernel_size/2 + (y+kernel_size/2)*kernel_size];

                    }
                    if(sum < 0)
                        sum = 0;
                    else
                        if(sum >255)
                            sum = 255;
                buffer[pixel] = sum;
                sum = 0;
            }

    end = clock(); 
    std::cout << "Time: " << (end - start) / CLOCKS_PER_SEC << " ms" << std::endl; 

    
    
    
    
    //Copy from buffer to image
    for(int i=0; i<width; i++)
        for(int j=0; j<height; j++)
            for(int k=0; k<=2; k++)
                output_image(i,j,k) = buffer[height*width*k + width*j + i];
    
    
    output_image.save("result.jpg");
        
        second_disp.display(output_image);
        int x = main_disp.window_x() + main_disp.window_width();
        second_disp.move(x, 0);
        main_disp.wait();
        second_disp.wait();
        
        delete [] buffer;
	
	/*
	while (!main_disp.is_closed() && !draw_disp.is_closed()) {
		main_disp.wait();
		if (main_disp.button() && main_disp.mouse_y()>=0) {
			const int y = main_disp.mouse_y();
			visu.fill(0).draw_graph(image.get_crop(0,y,0,0,image.width()-1,y,0,0),red,1,1,0,255,0);
			visu.draw_graph(image.get_crop(0,y,0,1,image.width()-1,y,0,1),green,1,1,0,255,0);
			visu.draw_graph(image.get_crop(0,y,0,2,image.width()-1,y,0,2),blue,1,1,0,255,0).display(draw_disp);
		}
	}
	 
	 */
	
	
	
	return 0;
}