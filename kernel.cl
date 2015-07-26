__kernel
void convolve (__global unsigned char * Md, __global unsigned char * Rd, __constant float * Kd, int width, int height, int kernel_size, int channels){

    int row = get_global_id(1);
    int col = get_global_id(0);
    
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
                    sum += Md[working_pixel] * Kd[x+kernel_size/2 + (y+kernel_size/2)*kernel_size];

                }
            Rd[pixel] = (int) sum;
            sum = 0;
        }
    }
}