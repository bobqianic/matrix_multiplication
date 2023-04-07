#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <immintrin.h>
#include <fmaintrin.h>

struct alignas(32) matrix_2D {
    float** body = nullptr;
    int x = 0;
    int y = 0;
};

matrix_2D get_rand_matrix_2D(int x, int y, float min, float max) {
    if (x * y == 0) {
        throw std::range_error("X and Y must > 0");
    }
    int new_x = x;
    int new_y = y;
    if (x % 8 != 0) {
        new_x += 8 - x % 8;
    }
    if (y % 8 != 0) {
        new_y += 8 - y % 8;
    }
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(min, max);
    matrix_2D matrix;
    matrix.x = new_x;
    matrix.y = new_y;
    matrix.body = new (std::align_val_t(32)) float* [matrix.x];
    for (int i = 0; i < matrix.x; i++) {
        matrix.body[i] = new (std::align_val_t(32)) float [matrix.y];
        for (int j = 0; j < matrix.y; j++) {
            if (j >= y or i >= x) {
                matrix.body[i][j] = 0;
            } else {
                matrix.body[i][j] = dist(mt);
            }
        }
    }
    return matrix;
}

matrix_2D get_matrix_2D(int x, int y, float value) {
    if (x * y == 0) {
        throw std::range_error("X and Y must > 0");
    }
    int new_x = x;
    int new_y = y;
    if (x % 8 != 0) {
        new_x += 8 - x % 8;
    }
    if (y % 8 != 0) {
        new_y += 8 - y % 8;
    }
    matrix_2D matrix;
    matrix.x = new_x;
    matrix.y = new_y;
    matrix.body = new (std::align_val_t(32)) float* [matrix.x];
    for (int i = 0; i < matrix.x; i++) {
        matrix.body[i] = new (std::align_val_t(32)) float [matrix.y];
        std::fill(matrix.body[i], matrix.body[i] + matrix.y, value);
    }
    return matrix;
}

template <typename T>
int print_matrix_2D(T a) {
    for (int y = 0; y < a.y; y++) {
        for (int x = 0; x < a.x; x++) {
            std::cout << a.body[x][y] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    return 0;
}

int free_matrix_2D(matrix_2D &a) {
    for (int x = 0; x < a.x; x++) {
        ::operator delete[] (a.body[x], std::align_val_t(32));
    }
    ::operator delete[] (a.body, std::align_val_t(32));
    return 0;
}

matrix_2D transpose_matrix_2D(matrix_2D &a) {
    matrix_2D matrix;
    matrix.x = a.y;
    matrix.y = a.x;
    matrix.body = new (std::align_val_t(32)) float* [matrix.x];
    for (int i = 0; i < matrix.x; i++) {
        matrix.body[i] = new (std::align_val_t(32)) float [matrix.y];
        for (int j = 0; j < matrix.y; j++) {
            matrix.body[i][j] = a.body[j][i];
        }
    }
    return matrix;
}

float sum8(__m256 x) {
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    const __m128 loQuad = _mm256_castps256_ps128(x);
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    const __m128 loDual = sumQuad;
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    const __m128 lo = sumDual;
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

matrix_2D matrix_mult_2D(matrix_2D &a, matrix_2D &b) {
    matrix_2D c = transpose_matrix_2D(a);
    matrix_2D d = get_matrix_2D(b.x, a.y, 0);
    for (int i = 0; i < a.y; i++) {
        for (int j = 0; j < b.x; j++) {
            __m256 seg_sum = _mm256_setzero_ps();
            for (int k = 0; k <= a.x - 8; k+=8) {
                __m256 seg_a = _mm256_load_ps(c.body[i] + k);
                __m256 seg_b = _mm256_load_ps(b.body[j] + k);
                seg_sum = _mm256_fmadd_ps(seg_a, seg_b, seg_sum);
            }
            d.body[j][i] = sum8(seg_sum);
        }
    }
    free_matrix_2D(c);
    return d;
}

int benchmark(int from, int to, int step) {
    auto* buffer = new float [1 + (to - from) / step];
    for (int size = from; size <= to; size += step) {
        auto in_1 = get_rand_matrix_2D(size, size, 0, 100);
        auto in_2 = get_rand_matrix_2D(size, size, 0, 100);
        auto start = std::chrono::high_resolution_clock::now();
        if (to <= 128) {
            for (int z = 0; z < 1000000; z++) {
                auto out_3 = matrix_mult_2D(in_1,in_2);
                free_matrix_2D(out_3);
            }
        } else if (to <= 512) {
            for (int z = 0; z < 100; z++) {
                auto out_3 = matrix_mult_2D(in_1,in_2);
                free_matrix_2D(out_3);
            }
        } else {
            for (int z = 0; z < 10; z++) {
                auto out_3 = matrix_mult_2D(in_1,in_2);
                free_matrix_2D(out_3);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        free_matrix_2D(in_1);
        free_matrix_2D(in_2);
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        if (to <= 128) {
            std::cout << size << "x" << size << " " << 1000000 * ((float)(2 * size - 1) * (float)(size * size)) / static_cast<float>(duration.count()) / 1000 << " GFLOPS" << std::endl;
            //std::cout << 1000000 * ((float)(2 * size - 1) * (float)(size * size)) / static_cast<float>(duration.count()) / 1000 << std::endl;
            buffer[(size - from) / step] = 1000000 * ((float)(2 * size - 1) * (float)(size * size)) / static_cast<float>(duration.count()) / 1000;
        } else if (to <= 512) {
            std::cout << size << "x" << size << " " << 100 * ((float)(2 * size - 1) * (float)(size * size)) / static_cast<float>(duration.count()) / 1000 << " GFLOPS" << std::endl;
            //std::cout << 100 * ((float)(2 * size - 1) * (float)(size * size)) / static_cast<float>(duration.count()) / 1000 << std::endl;
            buffer[(size - from) / step] = 100 * ((float)(2 * size - 1) * (float)(size * size)) / static_cast<float>(duration.count()) / 1000;
        } else {
            std::cout << size << "x" << size << " " << 10 * ((float)(2 * size - 1) * (float)(size * size)) / static_cast<float>(duration.count()) / 1000 << " GFLOPS" << std::endl;
            //std::cout << 10 * ((float)(2 * size - 1) * (float)(size * size)) / static_cast<float>(duration.count()) / 1000 << std::endl;
            buffer[(size - from) / step] = 10 * ((float)(2 * size - 1) * (float)(size * size)) / static_cast<float>(duration.count()) / 1000;
        }
    }
    std::ofstream file;
    file.open("data2.txt", std::ios::app);
    for (int i = 0; i < 1 + (to - from) / step; i++) {
        file << buffer[i] << std::endl;
    }
    file << "--------------------------" << std::endl;
    file.close();
    delete[] buffer;
    return 0;
}

int main() {
    auto in_1 = get_rand_matrix_2D(256, 256, 0, 100);
    auto in_2 = get_rand_matrix_2D(256, 256, 0, 100);
    auto start = std::chrono::high_resolution_clock::now();
    auto out_3 = matrix_mult_2D(in_1,in_2);
    auto end = std::chrono::high_resolution_clock::now();
    free_matrix_2D(in_1);
    free_matrix_2D(in_2);
    free_matrix_2D(out_3);
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (duration.count() < 1000) {
        std::cout << duration.count() << "us" << std::endl;
    } else if (duration.count() < 1000000){
        std::cout << static_cast<float>(duration.count()) / 1000 << "ms" << std::endl;
    } else {
        std::cout << static_cast<float>(duration.count()) / 1000000 << "s" << std::endl;
    }
    //SetProcessAffinityMask(GetCurrentProcess(), static_cast<DWORD_PTR>(1) << 1);
    //benchmark(8, 1024, 8);
    //benchmark(1088, 2048, 64);
    //benchmark(2304, 4096, 256);
}