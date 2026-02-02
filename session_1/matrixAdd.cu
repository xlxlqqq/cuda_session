#include <vector>

#include <stdlib.h>  // 用于 malloc 和 free
#include <stdio.h>
#include <stdexcept> // 用于 std::runtime_error

#include <cuda_runtime.h>

namespace CUDAMATRIX {

class CudaMemory {
public:
    
    CudaMemory(size_t size) {
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA Malloc Error: ") + cudaGetErrorString(err));
        }
    }

    ~CudaMemory() {
        cudaFree(ptr);
    }

    float* get() const {
        return static_cast<float*>(ptr);
    }

private:
    void* ptr;
};

    namespace Kernels {
        // kernal：矩阵加法
        __global__ void matrixAddKernel(float* A, float* B, float* C, int size) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < size) {
                C[index] = A[index] + B[index];
            }
        }
    }

    namespace Utils {

        // 检查是否存在可用的 CUDA 设备
        bool isCudaAvailable() {
            int deviceCount = 0;
            cudaError_t err = cudaGetDeviceCount(&deviceCount);
            return (err == cudaSuccess && deviceCount > 0);
        }

    };

    namespace Operations {
        // 矩阵加法
        std::vector<std::vector<float>> Add(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
            // 检查 CUDA 是否可用
            if (!Utils::isCudaAvailable()) {
                throw std::runtime_error("CUDA device not available");
            }

            if (A.size() != B.size() || A[0].size() != B[0].size()) {
                throw std::invalid_argument("Matrices dimensions do not match for addition");
            }
            
            int row = A.size();
            int col = A[0].size();
            int size = row * col;

            // 将二维矩阵展平成一维数组
            std::vector<float> hA(size), hB(size), hC(size);
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    hA[i * col + j] = A[i][j];
                    hB[i * col + j] = B[i][j];
                }
            }

            // 分配设备内存
            CudaMemory dA(size * sizeof(float));
            CudaMemory dB(size * sizeof(float));
            CudaMemory dC(size * sizeof(float));

            // 拷贝数据到设备
            cudaMemcpy(dA.get(), hA.data(), size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(dB.get(), hB.data(), size * sizeof(float), cudaMemcpyHostToDevice);

            // 启动 CUDA 核函数
            int threadsPerBlock = 256;
            int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
            Kernels::matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(dA.get(), dB.get(), dC.get(), size);
            
            try {
                cudaDeviceSynchronize();

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA Kernel Error: ") + cudaGetErrorString(err));
                }

                // 拷贝结果回主机
                err = cudaMemcpy(hC.data(), dC.get(), size * sizeof(float), cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    throw std::runtime_error("CUDA Memory Copy Back Failed");
                }
            } catch (...) {
                // 释放设备内存
                cudaFree(dA.get());
                cudaFree(dB.get());
                cudaFree(dC.get());
                throw;
            }
            
            cudaFree(dA.get());
            cudaFree(dB.get());
            cudaFree(dC.get());
            
            // 将一维结果转换回二维矩阵
            std::vector<std::vector<float>> C(row, std::vector<float>(col));
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    C[i][j] = hC[i * col + j];
                }
            }

            return C;
        }
    }

}  // namespace CUDAMATRIX

int main() {
    const int row = 16;
    const int col = 16;

    std::vector<std::vector<float>> A(row, std::vector<float>(col, 0.1f));
    std::vector<std::vector<float>> B(row, std::vector<float>(col, 0.2f));

    std::vector<std::vector<float>> result = CUDAMATRIX::Operations::Add(A, B);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", result[i][j]);
        }
        printf("\n");
    }

    return 0;
}
