#include <stdio.h>

__device__ float conv(float* matrix, float* kernal, int row, int col) {
    float sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int matrixRow = row + i;
            int matrixCol = col + j;
            sum += matrix[matrixRow * 5 + matrixCol] * kernal[i * 3 + j];
        }
    }
    return sum;
}

// 2D卷积核函数
__global__ void conv2D(float* matrix, float* kernal, float* output) {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int index = threadId + blockId * blockDim.x;

    int outputSize = 5 - 3 + 1; // 输入矩阵大小5x5，卷积核大小3x3
    if (index >= outputSize * outputSize) return;

    printf("Index: %d, blockId: %d, threadId: %d\n", index, blockId, threadId);

    // conv(matrix, kernal, blockId, threadId);
    output[index] = conv(matrix, kernal, blockId, threadId);
}

int main() {
    const int kernalSize = 3;
    const int matrixSize = 5;
    const int outputSize = matrixSize - kernalSize + 1;

    float* matrix = (float*)malloc(sizeof(float) * matrixSize * matrixSize);
    float* kernal = (float*)malloc(sizeof(float) * kernalSize * kernalSize);
    float* output = (float*)malloc(sizeof(float) * outputSize * outputSize);

    // 初始化矩阵和卷积核
    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            matrix[i * matrixSize + j] = 1.0f; // 示例值
        }
    }

    for (int i = 0; i < kernalSize; i++) {
        for (int j = 0; j < kernalSize; j++) {
            kernal[i * kernalSize + j] = 1.0f; // 示例值
        }
    }

    float* device_matrix;
    float* device_kernal;
    float* device_output;

    cudaMalloc((void**)&device_matrix, sizeof(float) * matrixSize * matrixSize);
    cudaMalloc((void**)&device_kernal, sizeof(float) * kernalSize * kernalSize);
    cudaMalloc((void**)&device_output, sizeof(float) * outputSize * outputSize);

    cudaMemcpy(device_matrix, matrix, sizeof(float) * matrixSize * matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(device_kernal, kernal, sizeof(float) * kernalSize * kernalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(device_output, output, sizeof(float) * outputSize * outputSize, cudaMemcpyHostToDevice);

    dim3 gridDim(outputSize);
    dim3 blockDim(outputSize);
    conv2D<<<gridDim, blockDim>>>(device_matrix, device_kernal, device_output);
    cudaDeviceSynchronize();

    cudaMemcpy(output, device_output, sizeof(float) * outputSize * outputSize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            printf("%f ", output[i * outputSize + j]);
        }
        printf("\n");
    }

    free(matrix);
    free(kernal);
    free(output);
    cudaFree(device_matrix);
    cudaFree(device_kernal);
    cudaFree(device_output);

    return 0;
}