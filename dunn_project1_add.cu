/*
 * CS-4370-90: Par. Prog. Many-Core GPUs
 * Nathan Dunn
 * Professor Liu
 * 10/4/19
 * Project 1 - Basic Matrix Addition
*/
#include <stdio.h>
#include <cuda.h>

#define N 8 // defines the size of matrix (i.e 8x8 16x16 etc...)

/**
	Performs vector addition on the GPU.
	dev_a - first matrix to be added
	dev_b - second matrix to be added
	dev_c - result of dev_a + dev_b stored in this matrix
	size - size of input matrices

*/
__global__ void gpuAdd(int *dev_a, int *dev_b, int *dev_c, int size){
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	int index = row * size + column;
	
	if(column < size && row < size){
		dev_c[index] = dev_a[index] + dev_b[index];
	}
}

/**
	Performs vector addition on the CPU.
	a - first matrix to be added
	b - second matrix to be added
	c - result of a + b stored in this matrix
	size - size of input matrices
*/
void cpuAdd(int *a, int *b, int *c, int size){
	
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			int index = i * size + j;
			c[index] = a[index] + b[index];
		}
	}
}

/**
	Prints a matrix.
	matrix - matrix to be printed
	size - size of the matrix
*/
void printMatrix(int * matrix, int size){
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			printf("%d ", matrix[i * size + j]);
		}
		printf("\n");
	}
	printf("\n");
}

/**
	Verifies that two matrices are equal.
	a - first matrix to be compared
	b - second matrix to be compared
	size - size of the matrix
*/
void verifySum(int *a, int *b, int size){
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			int index = i * size + j;
			if(a[index] != b[index]){
				goto FAILED;
			}
		}
	}
	
	printf("TEST PASSED!!!\n");
	return;
	
	FAILED: printf("TEST FAILED!!!\n");
}

int main(void){
	
	// define block size and count
	int blockSize = 4;
	int blockCount = ceil(N/double(blockSize)); 
	dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(blockCount, blockCount, 1);
	
	int *a, *b, *c, *d;
	
	int *dev_a, *dev_b, *dev_c;
	
	// allocate memory for matrix A, B
	a = (int*)malloc(sizeof(int)*N*N);
	b = (int*)malloc(sizeof(int)*N*N);
	c = (int*)malloc(sizeof(int)*N*N);
	d = (int*)malloc(sizeof(int)*N*N);
	
	// "fill" array with values
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			int index = i * N + j;
			a[index] = i + j;
			b[index] =  i - j;
		}
	}
	
	// perform CPU matrix addition for gpu addition verification
	cpuAdd(a, b, c, N);
	
	printf("Matrix A:\n");
	printMatrix(a, N);
	printf("\nMatrix B:\n");
	printMatrix(b, N);
	printf("\n CPU Sum of A + B:\n");
	printMatrix(c, N);
	
	// allocate device memory
	cudaMalloc((void **)(&dev_a), N*N*sizeof(int));
	cudaMalloc((void **)(&dev_b), N*N*sizeof(int));
	cudaMalloc((void **)(&dev_c), N*N*sizeof(int));
	
	// copy array a,b (system memory) to dev_a, dev_b (device memory)
	cudaMemcpy(dev_a,a,N*N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,N*N*sizeof(int), cudaMemcpyHostToDevice);
	
	printf("Thread Block Count: %d\n", blockCount);
	printf("Starting GPU Computations\n");
	
	// launch kernels
	gpuAdd<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);
	
	cudaDeviceSynchronize();
	// copy results from GPU back to system memory
	cudaMemcpy(d, dev_c, N*N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	printf("GPU Sum of A + B:\n");
	printMatrix(d, N);
	
	// verify that CPU and GPU addition match
	verifySum(c, d, N);
	
	// free system and device memory
	free(a);
	free(b);
	free(c);
	free(d); 
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	
	return 0;
}