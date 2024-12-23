#ifndef GENETIC_ALGORITHM_CUH
#define GENETIC_ALGORITHM_CUH

#include <cuda_runtime.h>

// Константы для генетического алгоритма
constexpr int POLYNOMIAL_DEGREE = 5; // Степень полинома
constexpr int INDIVIDUAL_SIZE = POLYNOMIAL_DEGREE + 1; // Размер одного индивидуала
constexpr int MAX_BLOCKS = 256; // Максимальное число блоков
constexpr int THREADS_PER_BLOCK = 256; // Число потоков на блок

// Структура для хранения точек
struct Point {
    float x;
    float y;
};

// Структура для параметров генетического алгоритма
struct GeneticParams {
    int populationSize;     // Размер популяции
    int maxGenerations;     // Максимальное число поколений
    float mutationRate;     // Вероятность мутации
    float crossoverRate;    // Вероятность скрещивания
    int numGenes;           // Число генов в одном индивидууме
    int degree;             // Степень полинома
};

// Функции, используемые в алгоритме
__device__ float evaluatePolynomial(const unsigned int* coefficients, float x);
__global__ void calculateFitness(const Point* points, int numPoints, const unsigned int* population, float* fitness);
__global__ void mutation(unsigned int* population, float mutationRate, int populationSize);
__global__ void crossover(unsigned int* population, float crossoverRate, int populationSize);

// Основная функция запуска алгоритма
void runGeneticAlgorithm(const Point* points, int numPoints, const GeneticParams& params, float* bestCoefficients);

#endif // GENETIC_ALGORITHM_CUH
