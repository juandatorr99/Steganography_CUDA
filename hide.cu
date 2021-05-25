//#include <bits/stdc++.h>
// #include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
// #include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define ThreadsPerBlock 32


__global__ void lessSignificativeBit(unsigned char *image, char *text, int textSize){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = col + row * blockDim.x * gridDim.x;
    
    int charno = offset / 8;
    if (charno >= textSize) {
        return;
    }
    int bit_count = 7 - (offset % 8);
    char ch = text[charno] >> bit_count;
    // if this bit is 1, then put 1 to the image RGB value, if bit == 0, put 0
    if (ch & 1) {
    image[offset] |= 1;
    } else {
    image[offset] &= ~1;
    }
}

int main(int argc, char **argv){
    // char *buff;
    char *textBuffer;
    int textFile;
    int long textSize;
    if(argc != 4){
        printf("Wrong number of arguments, usage: ./hide.out input.png text.txt output.png");
        exit(-1);
    }

    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        printf("There was an error trying to load the image");
        return -1;
    }

    if ( (textFile = open(argv[2], O_RDONLY)) < 0 ) {
		perror(argv[0]);
		return -2;
	}
    
    textSize = lseek(textFile, 0, SEEK_END);
    
    textBuffer = (char*) malloc(sizeof(char) * textSize);
    lseek(textFile, 0, SEEK_SET);
	read(textFile, textBuffer, sizeof(char) * textSize);
    
    printf("%s",textBuffer);
    
    int imageSize = image.step * image.rows;
    if((textSize+1)*8>imageSize*3){
        printf("The input text file is bigger than the maximum capacity of storage for the image");
        return -3;
    }
    //TextBuffer
    char *d_textBuffer;
    cudaMalloc((void **) &d_textBuffer, textSize * sizeof(char));
    cudaMemcpy(d_textBuffer,textBuffer, textSize * sizeof(char),cudaMemcpyHostToDevice);
    
    //Image
    cv::Mat output(image.rows, image.cols, CV_8UC3);
    unsigned char *d_image;
    cudaMalloc<unsigned char>(&d_image, imageSize);
    cudaMemcpy(d_image, image.ptr(), imageSize, cudaMemcpyHostToDevice);
    
    dim3 Blocks((image.cols + ThreadsPerBlock - 1) / ThreadsPerBlock,
                  (image.rows + ThreadsPerBlock - 1) / ThreadsPerBlock);
    dim3 Threads(ThreadsPerBlock,ThreadsPerBlock);
    
    lessSignificativeBit<<<Blocks, Threads>>>(d_image, d_textBuffer, textSize);
    
    
    cudaMemcpy(output.ptr(), d_image, imageSize, cudaMemcpyDeviceToHost);
    for(int i = textSize * 8; i<((textSize * 8)+8);i++){
         output.ptr()[i] &= ~1;
    }
    cv::imwrite(argv[3], output);
    
    //Close and Free
    cudaFree(d_image);
    cudaFree(d_textBuffer);
    free(textBuffer);
	close(textFile);

}