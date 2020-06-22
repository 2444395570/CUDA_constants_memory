#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <memory>


//������������
__constant__ int constant_f;
__constant__ int constant_g;
#define N 5


//�ں˺���Ϊ��ʹ�ó����ڴ�
__global__ void gpu_constant_memory(float* d_in, float* d_out) {
	//������ǰ�ں˵��߳�����
	int tid = threadIdx.x;
	d_out[tid] = constant_f * d_in[tid] + constant_g;
}

int main(void) {
	//Ϊ������������
	float h_in[N], h_out[N];
	//Ϊ�豸����ָ��
	float* d_in, * d_out;
	int h_f = 2;
	int h_g = 20;

	//��cpu�Ϸ����ڴ�
	cudaMalloc((void**)&d_in, N * sizeof(float));
	cudaMalloc((void**)&d_out, N * sizeof(float));

	//��ʼ������
	for (int i = 0; i < N; i++)
	{
		h_in[i] = i;
	}
	//�������������鵽�豸
	cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
	//���Ƴ����������ڴ�
	cudaMemcpyToSymbol(constant_f, &h_f, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(constant_g, &h_g, sizeof(int), 0, cudaMemcpyHostToDevice);

	//�����ں˴���һ�����N���߳�ÿ����
	gpu_constant_memory << <1, N >> > (d_in, d_out);

	//���豸�ڴ渴�ƽ�����ظ�����
	cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

	//��ӡ����������
	printf("Use of Constant memory on GPU\n");
	for (int i = 0; i < N; i++)
	{
		printf("The expression for index %f is %f\n", h_in[i], h_out[i]);
	}
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}
