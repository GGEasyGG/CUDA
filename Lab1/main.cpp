#include <cstdlib>
#include <cstdio>
#include <iostream>
 
using namespace std;
 
__global__ void mul_kernel(const double *a, const double *b, double *c, int n)
{ 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while(index < n)
    {
        c[index] = a[index] * b[index];
        index += offset;
    }
}
 
int main()
{
    int n;

    cin >> n;

    double *a_cpu = new double[n];
    double *b_cpu = new double[n];
    double *c_cpu = new double[n];

    double *a_gpu, *b_gpu, *c_gpu;

    cudaMalloc(&a_gpu, n * sizeof(double));
    cudaMalloc(&b_gpu, n * sizeof(double));
    cudaMalloc(&c_gpu, n * sizeof(double));
     
    for (int i = 0; i < n; i++)
    {
        cin >> a_cpu[i];
    }

    for (int i = 0; i < n; i++)
    {
        cin >> b_cpu[i];
    }
     
    cudaMemcpy(a_gpu, a_cpu, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu, n * sizeof(double), cudaMemcpyHostToDevice);
     
    mul_kernel <<< 256, 256 >>> (a_gpu, b_gpu, c_gpu, n);
     
    cudaMemcpy(c_cpu, c_gpu, n * sizeof(double), cudaMemcpyDeviceToHost);
     
    for(int i = 0; i < n; i++)
    {
        if(i != (n - 1))
        {
            printf("%.10e ", c_cpu[i]);
        }
        else:
        {
            printf("%.10e", c_cpu[i]);
        }
    }

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    delete [] a_cpu;
    delete [] b_cpu;
    delete [] c_cpu;
     
    return 0;
}
