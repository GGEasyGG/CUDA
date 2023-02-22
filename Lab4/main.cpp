#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <thrust\device_vector.h>
#include <thrust/extrema.h>

using namespace std;

__global__ void gauss_kernel(double* matrix, int n)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset_x = gridDim.x * blockDim.x;
    int offset_y = gridDim.y * blockDim.y;

    int max_index;
    double coeff;
    double tmp;

    double *row_array = (double *)malloc(sizeof(double) * n);

    for(int row = 0; row < n; row++)
    {
        for(int i = 0; i < n; i++)
        {
            row_array[i] = matrix[row * n + i];
        }

        thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(row_array);
        thrust::device_ptr<float> max_ptr = thrust::max_element(dev_ptr, dev_ptr + n);

        max_index = &max_ptr[0] - &dev_ptr[0];

        if(row != max_index)
        {
            for(int i = index_x; i < n; i += offset_x)
            {
                tmp = matrix[row * n + i];
                matrix[row * n + i] = matrix[max_index * n + i];
                matrix[max_index * n + i] = tmp;
            }
        }

        for(int i = row + 1 + index_x; i < n; i += offset_x)
        {
            coeff = matrix[i * n + row] / matrix[row * n + row];

            for(int j = row + 1 + index_y; j < n; j += offset_y)
            {
              matrix[i * n + j] -= coeff * matrix[row * n + j];
            }
        }

        for(int i = row + 1 + index_x; i < n; i += offset_x)
        {
            matrix[i * n + row] = 0.0;
        }
    }
}

int main()
{
    int n;

    cin >> n;

    double *matrix_cpu = new double[n * n];

    double *matrix_gpu;

    cudaMalloc(&matrix_gpu, n * n * sizeof(double));

    for (int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            cin >> matrix_cpu[i * n + j];
        }
    }

    cudaMemcpy(matrix_gpu, matrix_cpu, n * n * sizeof(double), cudaMemcpyHostToDevice);

    gauss_kernel <<< dim3(16, 16), dim3(32, 32) >>> (matrix_gpu, n);

    cudaMemcpy(matrix_cpu, matrix_gpu, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    double det = 1.0;

    for(int i = 0; i < n; i++)
    {
        det *= matrix_cpu[i * n + i];
    }

    printf("%.10e ", det);

    cudaFree(matrix_gpu);

    delete [] matrix_cpu;

    return 0;
}
