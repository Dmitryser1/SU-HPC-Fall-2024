#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// Размер блока для CUDA
#define BLOCK_SIZE 256

// Ядро CUDA для суммирования элементов вектора
__global__ void reduce_sum(float* input, float* output, int N) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    shared[tid] = (i < N) ? input[i] : 0;
    if (i + blockDim.x < N) {
        shared[tid] += input[i + blockDim.x];
    }

    __syncthreads();

    // Редукция внутри блока
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Записываем результат в глобальную память
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

// Функция для вычисления суммы на CPU
float sum_vector_cpu(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (float val : vec) {
        sum += val;
    }
    return sum;
}

// Функция для суммирования на GPU
float sum_vector_gpu(const std::vector<float>& vec) {
    int N = vec.size();
    int block_size = BLOCK_SIZE;
    int grid_size = (N + block_size * 2 - 1) / (block_size * 2);

    // Выделение памяти на GPU
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, grid_size * sizeof(float));

    // Копирование данных на GPU
    cudaMemcpy(d_input, vec.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Запуск ядра
    reduce_sum << <grid_size, block_size, block_size * sizeof(float) >> > (d_input, d_output, N);

    // Копирование результатов обратно на CPU
    std::vector<float> output(grid_size);
    cudaMemcpy(output.data(), d_output, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Финальная редукция на CPU
    float result = 0.0f;
    for (float val : output) {
        result += val;
    }

    // Очистка памяти
    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}

// Функция для замера времени выполнения
float measure_time(float (*func)(const std::vector<float>&), const std::vector<float>& vec) {
    clock_t start = clock();
    float result = func(vec);
    clock_t end = clock();
    double elapsed = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Результат: " << result << ", Время: " << elapsed << " секунд" << std::endl;
    return elapsed;
}

int main() {
    // Генерация вектора случайных чисел
    std::vector<float> vec(1'000'000);
    srand(time(0));
    for (float& val : vec) {
        val = static_cast<float>(rand()) / RAND_MAX;
    }

    // Суммирование на CPU
    std::cout << "CPU:" << std::endl;
    measure_time(sum_vector_cpu, vec);

    // Суммирование на GPU
    std::cout << "GPU:" << std::endl;
    measure_time(sum_vector_gpu, vec);

    return 0;
}
