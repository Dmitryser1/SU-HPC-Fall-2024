#include "genetic_algorithm.cuh"
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <thrust/device_vector.h>

// Установка генераторов случайных чисел
__global__ void setupRNG(curandState* state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

// Инициализация популяции
__global__ void initializePopulation(float* population, curandState* states, int populationSize, int numGenes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize * numGenes) {
        int geneIdx = idx % numGenes;
        population[idx] = curand_normal(&states[geneIdx]) * 0.1f; // Нормальное распределение
    }
}

// Вычисление приспособленности
__global__ void fitness_kernel(
    float* population,
    float* points_x,
    float* points_y,
    float* fitnesses,
    int populationSize,
    int numPoints,
    int degree
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;

    float error = 0.0f;
    for (int j = 0; j < numPoints; j++) {
        float x = points_x[j];
        float y = points_y[j];
        float approximation = 0.0f;

        for (int k = 0; k <= degree; k++) {
            approximation += population[idx * (degree + 1) + k] * powf(x, k);
        }

        error += powf(y - approximation, 2);
    }
    fitnesses[idx] = error;
}

// Мутация
__global__ void mutation_kernel(
    float* population,
    float* new_population,
    int populationSize,
    int numGenes,
    float mutationRate,
    curandState* states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize * numGenes) return;

    int individualIdx = idx / numGenes;
    int geneIdx = idx % numGenes;

    curandState localState = states[individualIdx];
    float gene = population[idx];

    if (curand_uniform(&localState) < mutationRate) {
        gene += curand_normal(&localState) * 0.1f;
    }

    new_population[idx] = gene;
    states[individualIdx] = localState;
}

// Основная функция запуска алгоритма
void runGeneticAlgorithm(
    const float* points_x,
    const float* points_y,
    int numPoints,
    const GeneticParams& params,
    float* bestCoefficients
) {
    // Выделение памяти на устройстве
    thrust::device_vector<float> d_population(params.populationSize * params.numGenes);
    thrust::device_vector<float> d_new_population(params.populationSize * params.numGenes);
    thrust::device_vector<float> d_fitnesses(params.populationSize);
    thrust::device_vector<float> d_points_x(points_x, points_x + numPoints);
    thrust::device_vector<float> d_points_y(points_y, points_y + numPoints);
    thrust::device_vector<curandState> d_states(params.populationSize);

    // Инициализация генераторов случайных чисел
    int blockSize = 256;
    int numBlocks = (params.populationSize + blockSize - 1) / blockSize;
    setupRNG<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_states.data()), time(nullptr));

    // Инициализация популяции
    initializePopulation<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_population.data()),
        thrust::raw_pointer_cast(d_states.data()),
        params.populationSize,
        params.numGenes
    );

    for (int generation = 0; generation < params.maxGenerations; generation++) {
        // Вычисление приспособленности
        fitness_kernel<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_population.data()),
            thrust::raw_pointer_cast(d_points_x.data()),
            thrust::raw_pointer_cast(d_points_y.data()),
            thrust::raw_pointer_cast(d_fitnesses.data()),
            params.populationSize,
            numPoints,
            params.degree
        );

        // Сортировка популяции по приспособленности
        thrust::sort_by_key(d_fitnesses.begin(), d_fitnesses.end(), d_population.begin());

        // Мутация
        mutation_kernel<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_population.data()),
            thrust::raw_pointer_cast(d_new_population.data()),
            params.populationSize,
            params.numGenes,
            params.mutationRate,
            thrust::raw_pointer_cast(d_states.data())
        );

        // Обновление популяции
        d_population.swap(d_new_population);
    }

    // Копирование лучших коэффициентов
    thrust::copy(
        d_population.begin(),
        d_population.begin() + params.numGenes,
        bestCoefficients
    );
}


int main() {
    // Пример входных данных
    const int numPoints = 5;
    float points_x[numPoints] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float points_y[numPoints] = {1.1f, 1.9f, 3.2f, 4.1f, 5.0f};

    GeneticParams params = {100, 1000, 0.05f, 0.7f, POLYNOMIAL_DEGREE + 1};

    // Массив для хранения лучших коэффициентов
    float bestCoefficients[POLYNOMIAL_DEGREE + 1] = {0.0f};

    try {
        runGeneticAlgorithm(points_x, points_y, numPoints, params, bestCoefficients);

        // Вывод лучших коэффициентов
        std::cout << "Best coefficients: ";
        for (int i = 0; i < POLYNOMIAL_DEGREE + 1; ++i) {
            std::cout << bestCoefficients[i] << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}


