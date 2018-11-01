/*
# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------
*/

#include <THC/THC.h>
#include "attentive_cuda_kernel.h"

extern THCState *state;

int attentive_conv(
    THCudaTensor *hidden_tensor,
    THCudaTensor *weight_tensor,
    THCudaTensor *attention_tensor,
    THCudaTensor *output_tensor,
    THCudaTensor *theta_tensor,
    THCudaTensor *output_k_tensor,
    THCudaTensor *output_kh_tensor,
    int k_c, int k_h, int k_w,
    int s_c, int s_h, int s_w,
    int p_c, int p_h, int p_w,
    int d_c, int d_h, int d_w,
		int group)
{
    int verbose = 0;

    float *hidden = THCudaTensor_data(state, hidden_tensor);
		float *weight = THCudaTensor_data(state, weight_tensor);
    float *attention = THCudaTensor_data(state, attention_tensor);
    float *output = THCudaTensor_data(state, output_tensor);
    float *theta = THCudaTensor_data(state, theta_tensor);
    float *output_k = THCudaTensor_data(state, output_k_tensor);
    float *output_kh = THCudaTensor_data(state, output_kh_tensor);
		cudaStream_t stream = THCState_getCurrentStream(state);

		int batch_size = THCudaTensor_size(state, hidden_tensor, 0);
		int hidden_c = THCudaTensor_size(state, hidden_tensor, 1);
		int hidden_h = THCudaTensor_size(state, hidden_tensor, 2);
		int hidden_w = THCudaTensor_size(state, hidden_tensor, 3);
		int attention_c = THCudaTensor_size(state, attention_tensor, 1);
		int attention_h = THCudaTensor_size(state, attention_tensor, 2);
		int attention_w = THCudaTensor_size(state, attention_tensor, 3);
		int output_c = THCudaTensor_size(state, output_tensor, 1);
		int output_h = THCudaTensor_size(state, output_tensor, 2);
		int output_w = THCudaTensor_size(state, output_tensor, 3);
		if(verbose==1){
      printf("%d\n", batch_size);
      printf("%d, %d, %d\n", hidden_c, hidden_h, hidden_w);
      printf("%d, %d, %d\n", attention_c, attention_h, attention_w);
      printf("%d, %d, %d\n", output_c, output_h, output_w);
    }

    attentive_conv_cuda(hidden, weight, attention, output, theta, output_k, output_kh,
		batch_size,
    hidden_c, hidden_h, hidden_w,
    attention_c, attention_h, attention_w,
    output_c, output_h, output_w,
    k_c, k_h, k_w,
    s_c, s_h, s_w,
    p_c, p_h, p_w,
    d_c, d_h, d_w,
		group,
    stream, verbose);

    return 1;
}


int attentive_pool(
    THCudaTensor *hidden_tensor,
    THCudaTensor *attention_tensor,
    THCudaTensor *output_tensor,
    THCudaTensor *ptable_tensor,
    int ptable_offset, int secondStageMode,
    int k_c, int k_h, int k_w,
    int s_c, int s_h, int s_w,
    int p_c, int p_h, int p_w,
    int d_c, int d_h, int d_w)
{
    int verbose = 0;

    float *hidden = THCudaTensor_data(state, hidden_tensor);
    float *attention = THCudaTensor_data(state, attention_tensor);
    float *output = THCudaTensor_data(state, output_tensor);
    float *ptable = THCudaTensor_data(state, ptable_tensor);
		cudaStream_t stream = THCState_getCurrentStream(state);

		int ptable_len = THCudaTensor_size(state, ptable_tensor, 0);
    int batch_size = THCudaTensor_size(state, hidden_tensor, 0);
		int hidden_c = THCudaTensor_size(state, hidden_tensor, 1);
		int hidden_h = THCudaTensor_size(state, hidden_tensor, 2);
		int hidden_w = THCudaTensor_size(state, hidden_tensor, 3);
		int attention_c = THCudaTensor_size(state, attention_tensor, 1);
		int attention_h = THCudaTensor_size(state, attention_tensor, 2);
		int attention_w = THCudaTensor_size(state, attention_tensor, 3);
		int output_c = THCudaTensor_size(state, output_tensor, 1);
		int output_h = THCudaTensor_size(state, output_tensor, 2);
		int output_w = THCudaTensor_size(state, output_tensor, 3);
		if(verbose==1){
      printf("%d\n", batch_size);
      printf("%d, %d, %d\n", hidden_c, hidden_h, hidden_w);
      printf("%d, %d, %d\n", attention_c, attention_h, attention_w);
      printf("%d, %d, %d\n", output_c, output_h, output_w);
    }

    attentive_pool_cuda(hidden, attention, output, ptable,
    ptable_len, ptable_offset, secondStageMode,
		batch_size,
    hidden_c, hidden_h, hidden_w,
    attention_c, attention_h, attention_w,
    output_c, output_h, output_w,
    k_c, k_h, k_w,
    s_c, s_h, s_w,
    p_c, p_h, p_w,
    d_c, d_h, d_w,
    stream, verbose);

    return 1;
}


int attentive_linear(
    THCudaTensor *hidden_tensor,
    THCudaTensor *weight_tensor,
    THCudaTensor *attention_tensor,
    THCudaTensor *output_tensor,
    THCudaTensor *theta_tensor,
    THCudaTensor *output_k_tensor,
    THCudaTensor *output_kh_tensor,
    THCudaTensor *ptable_tensor,
    int ptable_offset, int secondStageMode,
    int k_c, int k_h, int k_w,
    int s_c, int s_h, int s_w,
    int p_c, int p_h, int p_w,
    int d_c, int d_h, int d_w)
{
    int verbose = 0;

    float *hidden = THCudaTensor_data(state, hidden_tensor);
		float *weight = THCudaTensor_data(state, weight_tensor);
    float *attention = THCudaTensor_data(state, attention_tensor);
    float *output = THCudaTensor_data(state, output_tensor);
    float *theta = THCudaTensor_data(state, theta_tensor);
    float *output_k = THCudaTensor_data(state, output_k_tensor);
    float *output_kh = THCudaTensor_data(state, output_kh_tensor);
    float *ptable = THCudaTensor_data(state, ptable_tensor);
		cudaStream_t stream = THCState_getCurrentStream(state);

		int ptable_len = THCudaTensor_size(state, ptable_tensor, 0);
		int batch_size = THCudaTensor_size(state, hidden_tensor, 0);
		int hidden_c = THCudaTensor_size(state, hidden_tensor, 1);
		int hidden_h = THCudaTensor_size(state, hidden_tensor, 2);
		int hidden_w = THCudaTensor_size(state, hidden_tensor, 3);
		int attention_c = THCudaTensor_size(state, attention_tensor, 1);
		int attention_h = THCudaTensor_size(state, attention_tensor, 2);
		int attention_w = THCudaTensor_size(state, attention_tensor, 3);
		int output_c = THCudaTensor_size(state, output_tensor, 1);
		int output_h = THCudaTensor_size(state, output_tensor, 2);
		int output_w = THCudaTensor_size(state, output_tensor, 3);
		if(verbose==1){
      printf("%d\n", batch_size);
      printf("%d, %d, %d\n", hidden_c, hidden_h, hidden_w);
      printf("%d, %d, %d\n", attention_c, attention_h, attention_w);
      printf("%d, %d, %d\n", output_c, output_h, output_w);
    }

    attentive_linear_cuda(hidden, weight, attention, output, theta, output_k, output_kh, ptable,
    ptable_len, ptable_offset, secondStageMode,
		batch_size,
    hidden_c, hidden_h, hidden_w,
    attention_c, attention_h, attention_w,
    output_c, output_h, output_w,
    k_c, k_h, k_w,
    s_c, s_h, s_w,
    p_c, p_h, p_w,
    d_c, d_h, d_w,
    stream, verbose);

    return 1;
}
