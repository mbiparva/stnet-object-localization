/*
# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------
*/

#ifndef _ATTENTIVE_CONV_CUDA_KERNEL
#define _ATTENTIVE_CONV_CUDA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

void attentive_conv_cuda(
  float *hidden,
  float *weight,
	float *attention,
	float *output,
	float *theta,
  float *output_k,
  float *output_kh,
	int batch_size,
  int hidden_c, int hidden_h, int hidden_w,
  int attention_c, int attention_h, int attention_w,
  int output_c, int output_h, int output_w,
  int k_c, int k_h, int k_w,
  int s_c, int s_h, int s_w,
  int p_c, int p_h, int p_w,
  int d_c, int d_h, int d_w,
	int group,
  cudaStream_t stream, int verbose);

void attentive_pool_cuda(
  float *hidden,
	float *attention,
	float *output,
	float *ptable,
  int ptable_len, int ptable_offset, int secondStageMode,
	int batch_size,
  int hidden_c, int hidden_h, int hidden_w,
  int attention_c, int attention_h, int attention_w,
  int output_c, int output_h, int output_w,
  int k_c, int k_h, int k_w,
  int s_c, int s_h, int s_w,
  int p_c, int p_h, int p_w,
  int d_c, int d_h, int d_w,
  cudaStream_t stream, int verbose);

void attentive_linear_cuda(
  float *hidden,
  float *weight,
	float *attention,
	float *output,
	float *theta,
  float *output_k,
  float *output_kh,
	float *ptable,
  int ptable_len, int ptable_offset, int secondStageMode,
	int batch_size,
  int hidden_c, int hidden_h, int hidden_w,
  int attention_c, int attention_h, int attention_w,
  int output_c, int output_h, int output_w,
  int k_c, int k_h, int k_w,
  int s_c, int s_h, int s_w,
  int p_c, int p_h, int p_w,
  int d_c, int d_h, int d_w,
  cudaStream_t stream, int verbose);

#ifdef __cplusplus
}
#endif

#endif
