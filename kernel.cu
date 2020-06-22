#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <memory>


//定义两个常量
__constant__ int constant_f;
__constant__ int constant_g;
#define N 5


//内核函数为了使用常量内存
__global__ void gpu_constant_memory(float* d_in, float* d_out) {
	//给出当前内核的线程索引
	int tid = threadIdx.x;
	d_out[tid] = constant_f * d_in[tid] + constant_g;
}

int main(void) {
	//为主机定义数组
	float h_in[N], h_out[N];
	//为设备定义指针
	float* d_in, * d_out;
	int h_f = 2;
	int h_g = 20;

	//在cpu上分配内存
	cudaMalloc((void**)&d_in, N * sizeof(float));
	cudaMalloc((void**)&d_out, N * sizeof(float));

	//初始化数组
	for (int i = 0; i < N; i++)
	{
		h_in[i] = i;
	}
	//从主机复制数组到设备
	cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
	//复制常量到常量内存
	cudaMemcpyToSymbol(constant_f, &h_f, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(constant_g, &h_g, sizeof(int), 0, cudaMemcpyHostToDevice);

	//调用内核带有一个块和N个线程每个块
	gpu_constant_memory << <1, N >> > (d_in, d_out);

	//从设备内存复制结果返回给主机
	cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

	//打印结果在输出上
	printf("Use of Constant memory on GPU\n");
	for (int i = 0; i < N; i++)
	{
		printf("The expression for index %f is %f\n", h_in[i], h_out[i]);
	}
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}
