/*
# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------
*/

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
		int group);

int attentive_pool(
    THCudaTensor *hidden_tensor,
    THCudaTensor *attention_tensor,
    THCudaTensor *output_tensor,
    THCudaTensor *ptable_tensor,
    int ptable_offset, int secondStageMode,
    int k_c, int k_h, int k_w,
    int s_c, int s_h, int s_w,
    int p_c, int p_h, int p_w,
    int d_c, int d_h, int d_w);

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
    int d_c, int d_h, int d_w);
