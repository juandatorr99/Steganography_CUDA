#include <bits/stdc++.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define ThreadsPerBlock 16
using namespace std;
__global__ void lessSignificativeBit(unsigned char *image, char *text, int n){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = col + row * blockDim.x * gridDim.x;
    
    if (offset >= n) {
        return;
    }
    
    text[offset] = 0;
    //   if RGB value's last bit is 1, then we put 1 to the message, otherwise,
    //   use default 0
    if (image[offset] & 1) {
    text[offset] |= 1;
    }
    }

string BinaryStringToText(string binaryString) {
  string text = "";
  stringstream sstream(binaryString);
  while (sstream.good()) {
    bitset<8> bits;
    sstream >> bits;
    text += char(bits.to_ulong());
  }
  return text;
}

int main(int argc, char **argv){
    
    if (argc != 2) {
        printf("Wrong number of arguments");
        return -1;
    }
    
    
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        printf("Error trying to read the image");
        return -2;
    }
    
    int imageSize = image.step * image.rows;
    int n = image.cols * image.rows;
    
    unsigned char *d_image;
    char *d_text, *text;
    
    text = (char *)malloc(imageSize * sizeof(char));
    cudaMalloc((void **)&d_text, imageSize * sizeof(char));
    cudaMalloc<unsigned char>(&d_image, imageSize);
    
    cudaMemcpy(d_image, image.ptr(), imageSize, cudaMemcpyHostToDevice);
    
    dim3 Blocks((image.cols + ThreadsPerBlock - 1) / ThreadsPerBlock,(image.rows + ThreadsPerBlock - 1) / ThreadsPerBlock);
    dim3 Threads(ThreadsPerBlock,ThreadsPerBlock);
    
    
    lessSignificativeBit<<<Blocks, Threads>>>(d_image, d_text, n);
    
    
    cudaMemcpy(text, d_text, imageSize * sizeof(char),cudaMemcpyDeviceToHost);
    
    int i = 0, j = 0;
    while (i < imageSize - 8) {
    string oneChar = "";
    // add 8 bit to onechar, However, 0 cannot add to string, so convert to int first.
    for (j = 0; j < 8; j++) {
      int index = i + j;
      int num = (int)text[index];
      char temp[1];
      sprintf(temp, "%d", num);
      string s(temp);
      oneChar += s;
    }
    
    if (oneChar == "00000000") {
      break;
    }
    
    string ch = BinaryStringToText(oneChar);
    cout << ch;
    i += 8;
    }
    printf("\n");

}