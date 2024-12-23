#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// ������ ����� ��� CUDA
#define BLOCK_SIZE 256

// ���� CUDA ��� ������������ ��������� �������
__global__ void reduce_sum(float* input, float* output, int N) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    shared[tid] = (i < N) ? input[i] : 0;
    if (i + blockDim.x < N) {
        shared[tid] += input[i + blockDim.x];
    }

    __syncthreads();

    // �������� ������ �����
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // ���������� ��������� � ���������� ������
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

// ������� ��� ���������� ����� �� CPU
float sum_vector_cpu(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (float val : vec) {
        sum += val;
    }
    return sum;
}

// ������� ��� ������������ �� GPU
float sum_vector_gpu(const std::vector<float>& vec) {
    int N = vec.size();
    int block_size = BLOCK_SIZE;
    int grid_size = (N + block_size * 2 - 1) / (block_size * 2);

    // ��������� ������ �� GPU
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, grid_size * sizeof(float));

    // ����������� ������ �� GPU
    cudaMemcpy(d_input, vec.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // ������ ����
    reduce_sum << <grid_size, block_size, block_size * sizeof(float) >> > (d_input, d_output, N);

    // ����������� ����������� ������� �� CPU
    std::vector<float> output(grid_size);
    cudaMemcpy(output.data(), d_output, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    // ��������� �������� �� CPU
    float result = 0.0f;
    for (float val : output) {
        result += val;
    }

    // ������� ������
    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}

// ������� ��� ������ ������� ����������
float measure_time(float (*func)(const std::vector<float>&), const std::vector<float>& vec) {
    clock_t start = clock();
    float result = func(vec);
    clock_t end = clock();
    double elapsed = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "���������: " << result << ", �����: " << elapsed << " ������" << std::endl;
    return elapsed;
}

int main() {
    // ��������� ������� ��������� �����
    std::vector<float> vec(1'000'000);
    srand(time(0));
    for (float& val : vec) {
        val = static_cast<float>(rand()) / RAND_MAX;
    }

    // ������������ �� CPU
    std::cout << "CPU:" << std::endl;
    measure_time(sum_vector_cpu, vec);

    // ������������ �� GPU
    std::cout << "GPU:" << std::endl;
    measure_time(sum_vector_gpu, vec);

    return 0;
}
