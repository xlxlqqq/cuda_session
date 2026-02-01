#include <stdio.h>
#include <stdlib.h> // 用于 malloc 和 free

// 16 * 16 的矩阵相加
__global__ void matrixAdd(float* A, float* B, float* C, int N) {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int index = threadId + blockId * blockDim.x;

    if (index > 255) return;

    printf("Index: %d, blockId: %d, threadId: %d\n", index, blockId, threadId);
    C[index] = A[index] + B[index];

    return;
}

int main() {
    const int row = 16;
    const int col = 16;
    const int size = row * col;

    // 在主机上分配内存
    float* A = (float*)malloc(sizeof(float) * row * col);
    float* B = (float*)malloc(sizeof(float) * row * col);
    float* C = (float*)malloc(sizeof(float) * row * col);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            A[i * col + j] = 0.1f;
            B[i * col + j] = 0.2f;
        }
    }

    float* device_A;
    float* device_B;
    float* device_C;
    cudaMalloc((void**)&device_A, sizeof(float) * row * col);
    cudaMalloc((void**)&device_B, sizeof(float) * row * col);
    cudaMalloc((void**)&device_C, sizeof(float) * row * col);

    cudaMemcpy(device_A, A, sizeof(float) * row * col, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B, sizeof(float) * row * col, cudaMemcpyHostToDevice);
    
    dim3 gridDim(row);
    dim3 blockDim(col);
    matrixAdd<<<gridDim, blockDim>>>(device_A, device_B, device_C, row * col);
    cudaDeviceSynchronize();

    cudaMemcpy(C, device_C, sizeof(float) * row * col, cudaMemcpyDeviceToHost);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", C[i * col + j]);
        }
        printf("\n");
    }

    // 防止内存泄漏
    free(A);
    free(B);
    free(C);
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    return 0;
}