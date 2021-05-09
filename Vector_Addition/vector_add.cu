#include <iostream>
#include <cassert>
#include <time.h>

//Initializing CUDA kernel
//Called from CPU, runs in GPU
__global__ void vector_add(int *a, int *b, int *c, int n)
{
    //calculating globad tid
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //checking if the tid is not out of bounds
    if(tid<n)
        c[tid] = a[tid] + b[tid];
}

void verify_results(int *a, int *b, int *c, int n)
{
    //Asserting that the results calculated are correct
    for(int i=0; i<n; i++)
    {
        assert(c[i] == a[i] + b[i]);
    }
}

int main()
{
    //Performing operations for 65536 numbers
    int n = 1<<16;    

    //Pointers for CPU vectors  
    int *h_a, *h_b, *h_c;

    //Pointers for GPU vectors
    int *d_a, *d_b, *d_c;

    //Calculate memory needed for each vector
    size_t bytes = n*sizeof(int);

    //Allocate calculated memory on CPU or host
    h_a = (int *) malloc(bytes);
    h_b = (int *) malloc(bytes);
    h_c = (int *) malloc(bytes);

    //Allocate memory on GPU 
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    //Initializing arrays with random numbers
    for (int i=0; i<n; i++)
    {
        h_a[i] = rand();
        h_b[i] = rand();
    }

    //Copying arrays from CPU to GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    //No. of threads per block
    int num_threads = 1024;

    //No. of Thread Blocks
    int num_blocks = (int) ceil(float(n) / num_threads);

    //Starting time to calculate time taken on GPU
    clock_t start = clock();

    //Launch kernel on GPU
    vector_add<<<num_blocks, num_threads>>>(d_a, d_b, d_c, n);

    //Recording end time
    clock_t end = clock();

    //Copying results from GPU to CPU 
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    //Verifying results
    verify_results(h_a, h_b, h_c, n);

    //Free CUDA Memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //Free CPU Memory
    free(h_a);
    free(h_b);
    free(h_c);

    double time_taken = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Time Taken on GPU: " << time_taken << std::endl;
    std::cout << "Completed Successfully" << std::endl;

    return 0;
}