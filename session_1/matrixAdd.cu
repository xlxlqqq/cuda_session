#include <stdio.h>
#include <stdlib.h> // 用于 malloc 和 free
#include <vector>

// 16 * 16 的矩阵相加 kernal
__global__ void matrixAdd(float* A, float* B, float* C, int N) {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int index = threadId + blockId * blockDim.x;

    if (index >= N) return;

    // printf("Index: %d, blockId: %d, threadId: %d\n", index, blockId, threadId);
    C[index] = A[index] + B[index];
}

class MatrixAdd {

private:
    

    float* Add(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
        int row = A.size();
        int col = A[0].size();
        int size = row * col;

        std::vector<float> hA(size), hB(size);
        
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                hA[i * col + j] = A[i][j];
                hB[i * col + j] = B[i][j];
            }
        }

        float* result = (float*)malloc(sizeof(float) * size);

        float* device_A;
        float* device_B;
        float* device_C;
        cudaMalloc((void**)&device_A, sizeof(float) * size);
        cudaMalloc((void**)&device_B, sizeof(float) * size);
        cudaMalloc((void**)&device_C, sizeof(float) * size);

        cudaMemcpy(device_A, hA.data(), sizeof(float) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_B, hB.data(), sizeof(float) * size, cudaMemcpyHostToDevice);

        int threadNum = 256;
        int blockNum = (size + threadNum - 1) / threadNum;

        matrixAdd<<<blockNum, threadNum>>>(device_A, device_B, device_C, size);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(result, device_C, sizeof(float) * size, cudaMemcpyDeviceToHost);

        cudaFree(device_A);
        cudaFree(device_B);
        cudaFree(device_C);

        return result;
    }

public:
    MatrixAdd() = default; 
    ~MatrixAdd() = default;

    std::vector<std::vector<float>> Run(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
        if (A.size() != B.size() || A[0].size() != B[0].size()) {
            printf("Error: Matrix size not match\n");
            return std::vector<std::vector<float>>();
        }

        float* result = Add(A, B);
        std::vector<std::vector<float>> result_matrix(A.size(), std::vector<float>(A[0].size(), 0.0f));

        for (int i = 0; i < A.size(); i++) {
            for (int j = 0; j < A[0].size(); j++) {
                result_matrix[i][j] = result[i * A[0].size() + j];
            }
        }

        free(result);
        return result_matrix;
    }

};

int main() {
    const int row = 16;
    const int col = 16;
    // const int size = row * col;

    std::vector<std::vector<float>> A(row, std::vector<float>(col, 0.1f));
    std::vector<std::vector<float>> B(row, std::vector<float>(col, 0.2f));

    MatrixAdd matrixAdd;
    std::vector<std::vector<float>> result = matrixAdd.Run(A, B);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", result[i][j]);
        }
        printf("\n");
    }

    return 0;
}
