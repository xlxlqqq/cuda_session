#include <iostream>
#include <stdio.h>

// kernal函数只能用__global__修饰
// kernal只能用void返回值
__global__ void hello() {
  printf("Hello world from device.\n");
}

int main() {
  printf("Hello world from host.\n");

  hello<<<2,1>>>();
  cudaDeviceSynchronize();

  return 0;
}
