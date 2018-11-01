/*
# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------
*/

#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <assert.h>
// #include <time.h> 

#include "attentive_cuda_kernel.h"



// ****************************************************
// *************** DEVICE PORTION *********************
// ****************************************************
__device__ void scan(float *inputArray, float &cumSum, uint32_t remainer, int noThreads){
	assert ((remainer==0) || (remainer==1));
	const uint32_t blockSize = noThreads * 2 / (remainer + 1);
	const uint32_t thIdx = threadIdx.x;
	uint32_t offset = 1;
	float lastElem;

	// up sweep
	for (uint32_t upperBound=blockSize>>1; upperBound>0; upperBound>>=1){
		if(thIdx < upperBound)
			inputArray[offset*(2*thIdx+2)-1] += inputArray[offset*(2*thIdx+1)-1];
		offset <<= 1;
		__syncthreads();
	}

	// save last element and then set it with the identity.
	if (thIdx == 0){
		lastElem = inputArray[blockSize-1];
		inputArray[blockSize-1] = 0;
	}

	// down sweep
	for (uint32_t chunkSize=1; chunkSize<blockSize; chunkSize<<=1){
		offset >>= 1;
		__syncthreads();

		if(thIdx < chunkSize){
			uint32_t lowIdx  = offset*(2*thIdx+1) - 1;
			uint32_t highIdx = offset*(2*thIdx+2) - 1;

			float buffer = inputArray[lowIdx];
			inputArray[lowIdx] = inputArray[highIdx];
			inputArray[highIdx] += buffer;
		}
	}
	__syncthreads();

	// scatter increment of each block with last elem of the block before
	if (thIdx < blockSize>>1){
		inputArray[2*thIdx]   += cumSum;
		inputArray[2*thIdx+1] += cumSum;
	}
	__syncthreads();

	if (thIdx == 0)
		cumSum += lastElem;
}


__device__ inline void compAnge(float *fValue, float *sValue, bool tSortOrderAsc){
	if(tSortOrderAsc==true ? (*fValue > *sValue) : (*fValue < *sValue)){
		float buffer = *fValue;
		*fValue = *sValue;
		*sValue = buffer;
	}
}


__device__ void scan_fc(float *inputArray, float &cumSum, int noThreads){
	const uint32_t blockSize = noThreads * 2;
	const uint32_t thIdx = threadIdx.x;
	uint32_t offset = 1;
	float lastElem;

	// up sweep
	for (uint32_t upperBound=blockSize>>1; upperBound>0; upperBound>>=1){
		if(thIdx < upperBound)
			inputArray[offset*(2*thIdx+2)-1] += inputArray[offset*(2*thIdx+1)-1];
		offset <<= 1;
		__syncthreads();
	}

	// save last element and then set it with the identity.
	if (thIdx == 0){
		lastElem = inputArray[blockSize-1];
		inputArray[blockSize-1] = 0;
	}

	// down sweep
	for (uint32_t chunkSize=1; chunkSize<blockSize; chunkSize<<=1){
		offset >>= 1;
		__syncthreads();

		if(thIdx < chunkSize){
			uint32_t lowIdx  = offset*(2*thIdx+1) - 1;
			uint32_t highIdx = offset*(2*thIdx+2) - 1;

			float buffer = inputArray[lowIdx];
			inputArray[lowIdx] = inputArray[highIdx];
			inputArray[highIdx] += buffer;
		}
	}
	__syncthreads();

	// scatter increment of each block with last elem of the block before
	inputArray[2*thIdx]   += cumSum;
	inputArray[2*thIdx+1] += cumSum;
	__syncthreads();

	if (thIdx == 0)
		cumSum += lastElem;
}



// ********************************************************
// ********************** Conv ***************************
// *******************************************************
__global__ void firstStage(float *theta, float *hiddenBot, float *gatingTop, float *kernel, 
	float lowerBound,
	int noThreadsOrig, int noThreads,
	int kernelSizeOrig,
	int stride,
	int pad,
	int group,
	int dataLengthSOrig, int dataLength,
	int BOTsize,
	int sharedMemSize_1,
	int kernelLength){

	const uint32_t thIdx = threadIdx.x;
	const uint32_t blIdx = (blockIdx.x * gridDim.y * gridDim.z) + (blockIdx.y * gridDim.z) + (blockIdx.z);
	const uint32_t krIdx = (noThreadsOrig * kernelLength) * blockIdx.x;
	const int32_t rfIdxy = blockIdx.y * stride - pad;
	const int32_t rfIdxz = blockIdx.z * stride - pad;
	const uint32_t gPSIdxOrig = thIdx * kernelLength;
	const bool sortOrderAsc = true;
	const uint32_t groupIdx = (uint32_t) ((blockIdx.x * group) / gridDim.x);

	extern __shared__ float gatedPS[];		// dynamic shared memory
	__shared__ float gatingValTop;			// static shared memory
	__shared__ float psSum;
	__shared__ float cumSum;
	__shared__ bool found;
	if (thIdx == 0){
		gatingValTop = gatingTop[blIdx];
		psSum = 0.0f;
		cumSum = 0.0f;
		found = false;
	}
	__syncthreads();

	// exit if the value of the top gating node is zero.
	if (gatingValTop == 0)
		return;

	 // 0- Initialize the gatedPS in the shared memory by multiplying the kernel with the RF
	for (uint32_t krnCounterI = 0; krnCounterI < kernelSizeOrig; krnCounterI++){
		for (uint32_t krnCounterJ = 0; krnCounterJ < kernelSizeOrig; krnCounterJ++){
			if ((thIdx < noThreadsOrig) && (!((rfIdxy+(int32_t)krnCounterI<0) || (rfIdxy+(int32_t)krnCounterI>=BOTsize) || (rfIdxz+(int32_t)krnCounterJ<0) || (rfIdxz+(int32_t)krnCounterJ>=BOTsize))))		// check for the expansion paddind and the out of kernel boundary padding.
				gatedPS[gPSIdxOrig + krnCounterI*kernelSizeOrig+krnCounterJ] = kernel[krIdx+gPSIdxOrig+krnCounterI*kernelSizeOrig+krnCounterJ] *
																	hiddenBot[(((groupIdx*noThreadsOrig + thIdx)*BOTsize*BOTsize) + (uint32_t)rfIdxy*BOTsize + (uint32_t)rfIdxz)  +  krnCounterI*BOTsize+krnCounterJ];
			else
				gatedPS[gPSIdxOrig + krnCounterI*kernelSizeOrig+krnCounterJ] = 0.0f;		// here the padding value is zero using a constant strategy.
		}
	}
	__syncthreads();


	// 1- 1st Stage of the selection process
 	float localSum = 0.0f;
 	for (uint32_t psCounter = 0; psCounter<kernelLength; psCounter++)
 		localSum += gatedPS[gPSIdxOrig+psCounter];

	atomicAdd(&psSum, localSum);
	__syncthreads();


	if (psSum <= lowerBound){
		if (thIdx == 0)
			theta[blIdx] = -1000;
		return;
	}



	// BITONIC SORT: __Main loop over various subset sizes__
	for (uint32_t chunkSize = 1; chunkSize < dataLength; chunkSize <<= 1){
		for (uint32_t stride = chunkSize; stride > 0; stride >>= 1){
			for (uint32_t thCnt = thIdx; thCnt < dataLength>>1; thCnt += noThreads){
				uint32_t index  = thCnt;
				uint32_t offset = stride;
				if (stride == chunkSize){
					offset = 1 + ((index & (stride - 1)) << 1);
					index  = (index / stride) * stride  +  (stride - 1) - (index & (stride - 1));
				}
				index = (index << 1) - (index & (stride - 1));
				assert (index + offset < dataLength);

				if (index + offset < dataLengthSOrig)
					compAnge(&gatedPS[index], &gatedPS[index + offset], sortOrderAsc);
			}
			__syncthreads();
		}
	}


	// SUM SCAN:
	uint32_t remainer = kernelLength & 1;
	for (uint32_t gridCounter=0; gridCounter< kernelLength>>1; gridCounter++){
		scan(&gatedPS[noThreads*2*gridCounter], cumSum, 0, noThreads);
		__syncthreads();
	}
	if (remainer==1)
		scan(&gatedPS[noThreads*(kernelLength-remainer)], cumSum, remainer, noThreads);



	// find the theta according to the lowerBound.
	for (uint32_t thCnt = thIdx; thCnt < dataLengthSOrig-1; thCnt += noThreads){
		if (found)
			break;
		if ((gatedPS[thCnt] <= 0) && (gatedPS[thCnt+1] > 0)){
			theta[blIdx] = gatedPS[thCnt+1] - gatedPS[thCnt];
			found = true;
			break;
		}
	}
	__syncthreads();
	if ((thIdx==0) && !(found) && (gatedPS[dataLengthSOrig-1] <= 0) && (cumSum + lowerBound > 0)){
		theta[blIdx] = cumSum - gatedPS[dataLengthSOrig-1];
		found = true;
	}
	__syncthreads();
	if (!found)	
		for (uint32_t thCnt = thIdx; thCnt < dataLengthSOrig-1; thCnt += noThreads){
			if (found)
				break;
			if ((gatedPS[thCnt] == 0) && (gatedPS[thCnt+1] > 0)){
				theta[blIdx] = gatedPS[thCnt+1];
				found = true;
				break;
			}
		}
	__syncthreads();
	if (thIdx==0){
		if (!(found) && (gatedPS[dataLengthSOrig-1] == 0) && (cumSum > 0)){
			theta[blIdx] = cumSum - gatedPS[dataLengthSOrig-1];
			found = true;
		}

		if (!found && thIdx==0){
			printf("(%d, %d, %d), %f; %f; %f, %d \n", blockIdx.x, blockIdx.y, blockIdx.z, psSum, cumSum, gatingValTop, sharedMemSize_1);
			for (uint32_t thCnt = 0; thCnt < dataLengthSOrig; thCnt ++){
				printf("%d, %f\n", thCnt, gatedPS[thCnt]);
			}
		}
		assert (found);
	}
}

// ****************************************************
// ************* 1sr Stage - OverSized ****************
// ****************************************************
//This is called only if the kernel size is larger than hardware smem limit.
//First stage is re-implemented to be dynamic rahter than static.
__global__ void firstStage_oversized(float *theta, float *hiddenBot, float *gatingTop, float *kernel, 
	float lowerBound,
	int noThreadsOrig, int noThreads,
	int kernelSizeOrig,
	int stride,
	int pad,
	int group,
	int dataLengthSOrig, int dataLength,
	int BOTsize,
	int sharedMemSize_1,
	int kernelLength){

	const uint32_t thIdx = threadIdx.x;
	const uint32_t blIdx = (blockIdx.x * gridDim.y * gridDim.z) + (blockIdx.y * gridDim.z) + (blockIdx.z);
	const uint32_t krIdx = (noThreadsOrig * kernelLength) * blockIdx.x;
	const int32_t rfIdxy = blockIdx.y * stride - pad;
	const int32_t rfIdxz = blockIdx.z * stride - pad;
	const uint32_t gPSIdxOrig = thIdx * kernelLength;
	const bool sortOrderAsc = true;
	const uint32_t groupIdx = (uint32_t) ((blockIdx.x * group) / gridDim.x);
	const uint32_t noElemsOrig = dataLengthSOrig / noThreads;
	const float bins = 10;

	extern __shared__ float sharedMem[];		// dynamic shared memory
	float *smem_sort = (float*) sharedMem;
	float *smem_all  = (float*) &smem_sort[dataLengthSOrig];
	float *smem_neg  = (float*) &smem_all[noThreads];
	
	__shared__ float gatingValTop;			// static shared memory
	__shared__ float smem_non_pos;
	__shared__ float cumSum;
	__shared__ bool found;
	__shared__ float max_global;
	__shared__ float max_local;
	__shared__ float min_global;
	__shared__ float min_local;
	__shared__ bool range_set;
	__shared__ float denom_factor;

	if (thIdx == 0){
		gatingValTop = gatingTop[blIdx];
		smem_non_pos = 0.0f;
		cumSum = 0.0f;
		found = false;
		max_global = -1 * FLT_MAX;
		min_global = 0;
	}
	float local_buffer = 0;
	uint32_t local_counter = 0;
	uint32_t local_base_ind = 0;
	__syncthreads();

	// exit if the value of the top gating node is zero.
	if (gatingValTop == 0)
		return;

	// fill smem with zero
	smem_all[thIdx] = 0;
	smem_neg[thIdx] = 0;

	
	// 1- 1st Stage of the selection process
	if (thIdx < noThreadsOrig)
		for (uint32_t krnCounterI = 0; krnCounterI < kernelSizeOrig; krnCounterI++)
			for (uint32_t krnCounterJ = 0; krnCounterJ < kernelSizeOrig; krnCounterJ++){
				if (!((rfIdxy+(int32_t)krnCounterI<0) || (rfIdxy+(int32_t)krnCounterI>=BOTsize) || (rfIdxz+(int32_t)krnCounterJ<0) || (rfIdxz+(int32_t)krnCounterJ>=BOTsize)))		// check for the expansion paddind and the out of kernel boundary padding.
					local_buffer = kernel[krIdx+gPSIdxOrig+krnCounterI*kernelSizeOrig+krnCounterJ] * hiddenBot[(((groupIdx*noThreadsOrig + thIdx)*BOTsize*BOTsize) + (uint32_t)rfIdxy*BOTsize + (uint32_t)rfIdxz)  +  krnCounterI*BOTsize+krnCounterJ];
					smem_all[thIdx] += local_buffer;
					if (local_buffer < 0)
						smem_neg[thIdx] += local_buffer;
			}
	__syncthreads();

	// Sum Reduce: sum-reduce the shared memory using the half number of threads.
	for (uint32_t cnt = noThreads >> 1; cnt > 0; cnt >>= 1){
		if (thIdx < cnt){
			smem_all[thIdx] += smem_all[thIdx+cnt];
			smem_neg[thIdx] += smem_neg[thIdx+cnt];
		}
		__syncthreads();
	}
	if (smem_all[0] <= lowerBound){
		if (thIdx == 0)
			theta[blIdx] = -1000;
		return;
	}
	if (thIdx == 0)
		smem_non_pos = smem_neg[0];
	__syncthreads();


	// Set smem to zero
	smem_all[thIdx] = 0;
	smem_neg[thIdx] = 0;
	for (uint32_t i = 0; i < noElemsOrig; i++)
		smem_sort[thIdx + i] = 0;




	// MAIN BODY
	// find_max
	if (thIdx < noThreadsOrig){
		local_buffer = -1 * FLT_MAX;
		for (uint32_t krnCounterI = 0; krnCounterI < kernelSizeOrig; krnCounterI++)
			for (uint32_t krnCounterJ = 0; krnCounterJ < kernelSizeOrig; krnCounterJ++)
				if (!((rfIdxy+(int32_t)krnCounterI<0) || (rfIdxy+(int32_t)krnCounterI>=BOTsize) || (rfIdxz+(int32_t)krnCounterJ<0) || (rfIdxz+(int32_t)krnCounterJ>=BOTsize)))
					local_buffer = fmaxf(local_buffer, kernel[krIdx+gPSIdxOrig+krnCounterI*kernelSizeOrig+krnCounterJ] * hiddenBot[(((groupIdx*noThreadsOrig + thIdx)*BOTsize*BOTsize) + (uint32_t)rfIdxy*BOTsize + (uint32_t)rfIdxz)  +  krnCounterI*BOTsize+krnCounterJ]);
		smem_neg[thIdx] = local_buffer;
	}
	__syncthreads();
	// max reduce
	for (uint32_t cnt = noThreads >> 1; cnt > 0; cnt >>= 1){
		if (thIdx < cnt)
			smem_neg[thIdx] = fmaxf(smem_neg[thIdx], smem_neg[thIdx+cnt]);
		__syncthreads();
	}
	// assigne to the single shared variable.
	if (thIdx == 0)
		max_global = smem_neg[0];



	while(!found){
		// 0 - Initialize local max and min
		if (thIdx == 0){
			max_local = max_global;
			min_local = min_global;
			range_set = false;
		}
		__syncthreads();

		// 1 - Pivot Point Determination
		// set pivot
		while (!range_set){
			if (thIdx == 0){
				range_set = true;
				denom_factor = bins + 1;
			}
			__syncthreads();

			do {
				if (denom_factor < 2){
					if (thIdx == 0)
						range_set = false;
					__syncthreads();
					break;
				}

				if (thIdx == 0){
					denom_factor -= 1;
					max_local = min_local + (max_local - min_local) * denom_factor / bins;
				}
				__syncthreads();					

				// Counter
				smem_neg[thIdx] = 0;
				if (thIdx < noThreadsOrig)
					for (uint32_t krnCounterI = 0; krnCounterI < kernelSizeOrig; krnCounterI++)
						for (uint32_t krnCounterJ = 0; krnCounterJ < kernelSizeOrig; krnCounterJ++)
							if (!((rfIdxy+(int32_t)krnCounterI<0) || (rfIdxy+(int32_t)krnCounterI>=BOTsize) || (rfIdxz+(int32_t)krnCounterJ<0) || (rfIdxz+(int32_t)krnCounterJ>=BOTsize))){
								local_buffer = kernel[krIdx+gPSIdxOrig+krnCounterI*kernelSizeOrig+krnCounterJ] * hiddenBot[(((groupIdx*noThreadsOrig + thIdx)*BOTsize*BOTsize) + (uint32_t)rfIdxy*BOTsize + (uint32_t)rfIdxz)  +  krnCounterI*BOTsize+krnCounterJ];
								if ((local_buffer > min_local) && (local_buffer <= max_local))
									smem_neg[thIdx] += 1;
							}
				smem_all[thIdx] = smem_neg[thIdx];
				__syncthreads();
				// sum reduce
				for (uint32_t cnt = noThreads >> 1; cnt > 0; cnt >>= 1){
					if (thIdx < cnt)
						smem_neg[thIdx] += smem_neg[thIdx+cnt];
					__syncthreads();
				}			
			}
			while(smem_neg[0] > dataLengthSOrig);
		}
		
		// 2 - Find and Fill
		if (thIdx == 0)
			cumSum = 0.0f;
		scan(smem_all, cumSum, 0, noThreads/2);
		__syncthreads();
		//sanity check
		assert(smem_neg[0] == cumSum);

		local_counter = 0;
		local_base_ind = (uint32_t) smem_all[thIdx];
		if (thIdx == noThreads - 1)
			smem_neg[thIdx] = cumSum - local_base_ind;
		else
			smem_neg[thIdx] = smem_all[thIdx+1] - local_base_ind;
		if ((thIdx < noThreadsOrig) && (smem_neg[thIdx] != 0)){
			for (uint32_t krnCounterI = 0; krnCounterI < kernelSizeOrig; krnCounterI++)
				for (uint32_t krnCounterJ = 0; krnCounterJ < kernelSizeOrig; krnCounterJ++)
					if (!((rfIdxy+(int32_t)krnCounterI<0) || (rfIdxy+(int32_t)krnCounterI>=BOTsize) || (rfIdxz+(int32_t)krnCounterJ<0) || (rfIdxz+(int32_t)krnCounterJ>=BOTsize))){	
						local_buffer = kernel[krIdx+gPSIdxOrig+krnCounterI*kernelSizeOrig+krnCounterJ] * hiddenBot[(((groupIdx*noThreadsOrig + thIdx)*BOTsize*BOTsize) + (uint32_t)rfIdxy*BOTsize + (uint32_t)rfIdxz)  +  krnCounterI*BOTsize+krnCounterJ];
						if ((local_buffer > min_local) && (local_buffer <= max_local)){
							smem_sort[local_base_ind + local_counter] = local_buffer;
							local_counter += 1;
						}
					}
		}
		__syncthreads();


		// 3 - Sort smem
		for (uint32_t chunkSize = 1; chunkSize < dataLength; chunkSize <<= 1){
			for (uint32_t stride = chunkSize; stride > 0; stride >>= 1){
				for (uint32_t thCnt = thIdx; thCnt < dataLength>>1; thCnt += noThreads){
					uint32_t index  = thCnt;
					uint32_t offset = stride;
					if (stride == chunkSize){
						offset = 1 + ((index & (stride - 1)) << 1);
						index  = (index / stride) * stride  +  (stride - 1) - (index & (stride - 1));
					}
					index = (index << 1) - (index & (stride - 1));
					assert (index + offset < dataLength);

					if (index + offset < dataLengthSOrig)
						compAnge(&smem_sort[index], &smem_sort[index + offset], sortOrderAsc);
				}
				__syncthreads();
			}
		}

		// 4 - Find Theta
		if (thIdx == 0){
			for (uint32_t cnt = 0; cnt < dataLengthSOrig; cnt++){
				if (smem_sort[cnt] <= min_local)
					continue;
				smem_non_pos += smem_sort[cnt];
				if (smem_non_pos > lowerBound){
					theta[blIdx] = smem_sort[cnt];
					found = true;
					break;
				}
			}
			min_global = max_local;
		}	
		
		__syncthreads(); 
	}
}

// ******************sum scan algorithm**********************************
__device__ inline void swipeChange(uint8_t *array, uint8_t newVal, uint8_t oldVal, uint8_t labelCounter){
	while (newVal ^ array[newVal])	// climb the tree to the top root.
		newVal = array[newVal];

	for (uint8_t i=1; i<=labelCounter; i++)		// update all similar cells.
		if (array[i] == oldVal)
			array[i] = newVal;
}

__device__ inline uint32_t modOpt(uint32_t input, uint32_t modOf){
	if (modOf == 2){
		return input & (modOf - 1);
	}
	if (modOf == 3){		// maximum value is 9-1 beginning from 0
		assert (input < 12);
		if (input >= 6) input -= 6;
		if (input >= 3) input -= 3;
		return input;
	}
	else if (modOf == 5){	// maximum value is 25-1 beginning from 0
		assert (input < 40);
		if (input >= 20) input -= 20;
		if (input >= 10) input -= 10;
		if (input >=  5) input -=  5;
		return input;
	}
	else if (modOf == 6){	// maximum value is 36-1 beginning from 0
		assert (input < 48);
		if (input >= 24) input -= 24;
		if (input >= 12) input -= 12;
		if (input >=  6) input -=  6;
		return input;
	}
	else if (modOf == 7){	// maximum value is 49-1 beginning from 0
		assert (input < 56);
		if (input >= 28) input -= 28;
		if (input >= 14) input -= 14;
		if (input >=  7) input -=  7;
		return input;
	}
	else if (modOf == 11){	// maximum value is 121-1 beginning from 0
		assert (input < 176);
		if (input >= 88) input -= 88;
		if (input >= 44) input -= 44;
		if (input >= 22) input -= 22;
		if (input >= 11) input -= 11;
		return input;
	}
	else
		return NULL;
}

__global__ void secondStage(float *gatingBot, float *thetaIn, float *hiddenBot, float *gatingTop, float *kernel, float *kernelGating, float *kernelGatingHit, float *abnBelief,
	float epsilon,
	int noThreadsOrig,
	int kernelSizeOrig,
	int stride,
	int pad,
	int group,
	int BOTsize,
	int kernelLength,
	int smem_2_oversized,
	int reduceInitSize,
	int memBankRem,
	int ABNSwitch,
	float EBMult,
	int KernelGatingSwitch){

	assert ((kernelSizeOrig == 3) || (kernelSizeOrig == 5) || (kernelSizeOrig == 6) || (kernelSizeOrig == 7) || (kernelSizeOrig == 11));
	// Constant memory
	const uint8_t MAXNoCCs = 6*6;			// for the max 11x11 2D kernel, 36 maximum CC could exist.
	const uint32_t thdIdx = threadIdx.x;
	const uint32_t thdOffset = thdIdx * kernelLength;
	const uint32_t blIdx = (blockIdx.x * gridDim.y * gridDim.z) + (blockIdx.y * gridDim.z) + (blockIdx.z);
	const uint32_t krIdx = (blockDim.x * kernelLength) * blockIdx.x;
	const int32_t rfIdxy = blockIdx.y * stride - pad;
	const int32_t rfIdxz = blockIdx.z * stride - pad;
	const uint32_t groupIdx = (uint32_t) ((blockIdx.x * group) / gridDim.x);
	const float abnBeliefThd = (ABNSwitch==1) ? abnBelief[groupIdx*noThreadsOrig + thdIdx] : 1.0f;

	// Local memory
	uint8_t equivArray[MAXNoCCs];
	float labelPSWSum[MAXNoCCs] = {[0 ... MAXNoCCs-1] = 0.0f};
	uint8_t labelAreaSum[MAXNoCCs] = {[0 ... MAXNoCCs-1] = 0};

	// Shared memory
	extern __shared__ float sharedMem[];	// dynamic shared memory
	float *gatedPS;
	uint8_t *labelArray;
	float *maxArray;

	if (smem_2_oversized == 0){
		gatedPS  	= (float*) sharedMem;
		labelArray  = (uint8_t*) &gatedPS[blockDim.x * kernelLength];
		maxArray 	= (float*) &labelArray[blockDim.x * kernelLength + memBankRem];
	}
	else {
		labelArray  = (uint8_t*) sharedMem;
		maxArray 	= (float*) &labelArray[blockDim.x * kernelLength + memBankRem];
	}

	__shared__ float gatingValTop;			// static shared memory
	__shared__ float theta;
	__shared__ bool labelArrayNZ;
	if (thdIdx == 0){
		gatingValTop = gatingTop[blIdx];
		theta = thetaIn[blIdx];
		labelArrayNZ = false;
	}
	__syncthreads();


	// exit if the value of the top gating node is zero.
	if (gatingValTop == 0 || theta <= 0)
		return;
	assert (theta > 0);

	if (smem_2_oversized == 0){
		// 0) Initialize the gatedPS in the shared memory by multiplying the kernel with the RF
		for (uint32_t krnCounterI = 0; krnCounterI < kernelSizeOrig; krnCounterI++){
			for (uint32_t krnCounterJ = 0; krnCounterJ < kernelSizeOrig; krnCounterJ++){
				if (!((rfIdxy+(int32_t)krnCounterI<0) || (rfIdxy+(int32_t)krnCounterI>=BOTsize) || (rfIdxz+(int32_t)krnCounterJ<0) || (rfIdxz+(int32_t)krnCounterJ>=BOTsize)))
					gatedPS[thdOffset + krnCounterI*kernelSizeOrig+krnCounterJ] = kernel[krIdx+thdOffset+krnCounterI*kernelSizeOrig+krnCounterJ] *
																			hiddenBot[(((groupIdx*blockDim.x + thdIdx)*BOTsize*BOTsize) + (uint32_t)rfIdxy*BOTsize + (uint32_t)rfIdxz)  +  krnCounterI*BOTsize+krnCounterJ];
				else
					gatedPS[thdOffset + krnCounterI*kernelSizeOrig+krnCounterJ] = 0.0f;
			}
		}
		__syncthreads();

		float numericFactor = 0.0f;
		while (!labelArrayNZ){
			numericFactor+=1.0f;
			__syncthreads();
			for (uint32_t i=0; i<kernelLength; i++){
				if (gatedPS[thdOffset + i] >= theta - (epsilon*numericFactor)){
					labelArray[thdOffset + i] = 1;
					if (!labelArrayNZ)
						labelArrayNZ = true;
				}
				else
					labelArray[thdOffset + i] = 0;
			}
			__syncthreads();
		}
		if (!labelArrayNZ){
			printf("%f, %f,\n", gatingValTop, theta);
			for (uint32_t i=0; i<kernelLength; i++)
				printf("%d,%d: %f\n", thdIdx,i, gatedPS[thdOffset + i]);
		}
	}
	else {
		float numericFactor = 0.0f;
		while (!labelArrayNZ){
			numericFactor+=1.0f;
			__syncthreads();
			for (uint32_t krnCounterI = 0; krnCounterI < kernelSizeOrig; krnCounterI++){
				for (uint32_t krnCounterJ = 0; krnCounterJ < kernelSizeOrig; krnCounterJ++){
					labelArray[thdOffset + krnCounterI*kernelSizeOrig+krnCounterJ] = 0;
					if (!((rfIdxy+(int32_t)krnCounterI<0) || (rfIdxy+(int32_t)krnCounterI>=BOTsize) || (rfIdxz+(int32_t)krnCounterJ<0) || (rfIdxz+(int32_t)krnCounterJ>=BOTsize))) {
						if (kernel[krIdx+thdOffset+krnCounterI*kernelSizeOrig+krnCounterJ] * hiddenBot[(((groupIdx*blockDim.x + thdIdx)*BOTsize*BOTsize) + (uint32_t)rfIdxy*BOTsize + (uint32_t)rfIdxz)  +  krnCounterI*BOTsize+krnCounterJ] >= theta - (epsilon*numericFactor)){
							labelArray[thdOffset + krnCounterI*kernelSizeOrig+krnCounterJ] = 1;
							if (!labelArrayNZ) 
								labelArrayNZ = true;
						}
					}
				}
			}
			__syncthreads();
		}		
		if (!labelArrayNZ){
			printf("%f, %f,\n", gatingValTop, theta);
			printf("in smem_2_oversized block, all label array is zero for thread index %d\n", thdIdx);
		}
	}
	assert (labelArrayNZ);
	


	// 1) First Pass
	bool hasNeighbor;
	uint8_t minLabel;
	int32_t neighborIdx;
	uint32_t lrCol;
	uint8_t labelCounter = 0;
	for (uint32_t i=0; i<kernelLength; i++){
		if (labelArray[thdOffset + i] == 0)
			continue;

		// boundary conditions
		hasNeighbor = false;
		minLabel = kernelLength;
		lrCol = modOpt(i, kernelSizeOrig);

		neighborIdx = i - 1;		// West
		if ((lrCol) && (labelArray[thdOffset + neighborIdx] != 0)){
			hasNeighbor = true;
			minLabel = min(minLabel, labelArray[thdOffset + neighborIdx]);
		}
		neighborIdx -= kernelSizeOrig;	// West-North
		if ((lrCol) && (neighborIdx >= 0) && (labelArray[thdOffset + neighborIdx] != 0)){
			hasNeighbor = true;
			minLabel = min(minLabel, labelArray[thdOffset + neighborIdx]);
		}
		neighborIdx++;				// North
		if ((neighborIdx >= 0) && (labelArray[thdOffset + neighborIdx] != 0)){
			hasNeighbor = true;
			minLabel = min(minLabel, labelArray[thdOffset + neighborIdx]);
		}
		neighborIdx++;				// North-East
		if ((lrCol+1 != kernelSizeOrig) && (neighborIdx >= 0) && (labelArray[thdOffset + neighborIdx] != 0)){
			hasNeighbor = true;
			minLabel = min(minLabel, labelArray[thdOffset + neighborIdx]);
		}

		if (hasNeighbor)
			labelArray[thdOffset + i] = minLabel;
		else
			labelArray[thdOffset + i] = ++labelCounter;
	}



	// 2) Initialize the equivalence array with indeces.
	for (uint8_t cnt = 0; cnt < MAXNoCCs; cnt++)
		equivArray[cnt] = cnt;

	// 3) Update the equivalence array according to the updates labels.
	for (uint32_t i=0; i<kernelLength; i++){
		if (labelArray[thdOffset + i] == 0)
			continue;
		lrCol = modOpt(i, kernelSizeOrig);
		neighborIdx = i - 1;		// West
		if ((lrCol) && (labelArray[thdOffset + neighborIdx] != 0) && (labelArray[thdOffset + i] != labelArray[thdOffset + neighborIdx]) && (labelArray[thdOffset + i] != equivArray[labelArray[thdOffset + neighborIdx]]))
			swipeChange(equivArray, labelArray[thdOffset + i], equivArray[labelArray[thdOffset + neighborIdx]], labelCounter);
		neighborIdx -= kernelSizeOrig;	// West-North
		if ((lrCol) && (neighborIdx >= 0) && (labelArray[thdOffset + neighborIdx] != 0) && (labelArray[thdOffset + i] != labelArray[thdOffset + neighborIdx]) && (labelArray[thdOffset + i] != equivArray[labelArray[thdOffset + neighborIdx]]))
			swipeChange(equivArray, labelArray[thdOffset + i], equivArray[labelArray[thdOffset + neighborIdx]], labelCounter);
		neighborIdx++;				// North
		if ((neighborIdx >= 0) && (labelArray[thdOffset + neighborIdx] != 0) && (labelArray[thdOffset + i] != labelArray[thdOffset + neighborIdx]) && (labelArray[thdOffset + i] != equivArray[labelArray[thdOffset + neighborIdx]]))
			swipeChange(equivArray, labelArray[thdOffset + i], equivArray[labelArray[thdOffset + neighborIdx]], labelCounter);
		neighborIdx++;				// North-East
		if ((lrCol+1 != kernelSizeOrig) && (neighborIdx >= 0) && (labelArray[thdOffset + neighborIdx] != 0) && (labelArray[thdOffset + i] != labelArray[thdOffset + neighborIdx]) && (labelArray[thdOffset + i] != equivArray[labelArray[thdOffset + neighborIdx]]))
			swipeChange(equivArray, labelArray[thdOffset + i], equivArray[labelArray[thdOffset + neighborIdx]], labelCounter);
	}

	// 4) Update the label array according to the equivalence array
	for (uint32_t i=0; i<kernelLength; i++){
		if (labelArray[thdOffset + i] == 0)
			continue;
		labelArray[thdOffset + i] = equivArray[labelArray[thdOffset + i]];
	}


	// 6) compute the PSW and Area over remained labels
	if (smem_2_oversized == 0){
		for (uint32_t i=0; i<kernelLength; i++){
			if (labelArray[thdOffset + i] == 0)
				continue;
			labelAreaSum[labelArray[thdOffset + i]]++;
			labelPSWSum[labelArray[thdOffset + i]] += gatedPS[thdOffset + i];
		}
	}
	else {
		for (uint32_t krnCounterI = 0; krnCounterI < kernelSizeOrig; krnCounterI++){
			for (uint32_t krnCounterJ = 0; krnCounterJ < kernelSizeOrig; krnCounterJ++){
				if (labelArray[thdOffset + krnCounterI*kernelSizeOrig+krnCounterJ] == 0) continue;
				labelAreaSum[labelArray[thdOffset + krnCounterI*kernelSizeOrig+krnCounterJ]]++;
				if (!((rfIdxy+(int32_t)krnCounterI<0) || (rfIdxy+(int32_t)krnCounterI>=BOTsize) || (rfIdxz+(int32_t)krnCounterJ<0) || (rfIdxz+(int32_t)krnCounterJ>=BOTsize)))
					labelPSWSum[labelArray[thdOffset + krnCounterI*kernelSizeOrig+krnCounterJ]] += kernel[krIdx+thdOffset+krnCounterI*kernelSizeOrig+krnCounterJ] * hiddenBot[(((groupIdx*blockDim.x + thdIdx)*BOTsize*BOTsize) + (uint32_t)rfIdxy*BOTsize + (uint32_t)rfIdxz)  +  krnCounterI*BOTsize+krnCounterJ];
				else {
					printf("this must never happen due to the top if condition, report to the developer\n");
					assert(false);
				}
			}
		}
	}

	// 7) find the maximum label according to the mixed formula;
	float fusedVal = 0.0f;
	float maxFusedVal = 0.0f;
	uint8_t maxFusedIdx = 0;
	for (uint8_t i=1; i<=labelCounter; i++){
		if (labelAreaSum[i] == 0)
			continue;
		fusedVal = abnBeliefThd * EBMult * labelPSWSum[i] + (1.0f - EBMult) * (float) labelAreaSum[i];
		maxFusedVal = max(fusedVal, maxFusedVal);
		if (maxFusedVal == fusedVal)
			maxFusedIdx = i;
	}

	// 8) copy the max fused value into the shared memory
	maxArray[thdIdx] = maxFusedVal;
	__syncthreads();


	// 9) reduce the max array shared memory using the half number of threads.
	for (uint32_t cnt = reduceInitSize>>1; cnt > 0; cnt >>= 1){
		if (thdIdx < cnt && thdIdx+cnt < blockDim.x && maxArray[thdIdx] < maxArray[thdIdx+cnt])
			maxArray[thdIdx] = maxArray[thdIdx+cnt];
		__syncthreads();
	}

	assert (maxArray[0] != 0);


	// 10) Normalize and weight sum the gatingBot nodes.
	if (maxFusedVal == maxArray[0]){
		float gatedPS_local = 0.0;
		for (uint32_t krnCounterI = 0; krnCounterI < kernelSizeOrig; krnCounterI++){
			for (uint32_t krnCounterJ = 0; krnCounterJ < kernelSizeOrig; krnCounterJ++){
				uint32_t kernelIdx = thdOffset + krnCounterI * kernelSizeOrig + krnCounterJ;
				if ((labelArray[kernelIdx] == maxFusedIdx) && !((rfIdxy+(int32_t)krnCounterI<0) || (rfIdxy+(int32_t)krnCounterI>=BOTsize) || (rfIdxz+(int32_t)krnCounterJ<0) || (rfIdxz+(int32_t)krnCounterJ>=BOTsize))){
					if (smem_2_oversized == 0) gatedPS_local = gatedPS[kernelIdx];
					else gatedPS_local = kernel[krIdx+thdOffset+krnCounterI*kernelSizeOrig+krnCounterJ] * hiddenBot[(((groupIdx*blockDim.x + thdIdx)*BOTsize*BOTsize) + (uint32_t)rfIdxy*BOTsize + (uint32_t)rfIdxz)  +  krnCounterI*BOTsize+krnCounterJ];
					atomicAdd(&gatingBot[(((groupIdx*blockDim.x + thdIdx)*BOTsize*BOTsize) + (uint32_t)rfIdxy*BOTsize + (uint32_t)rfIdxz)  +  krnCounterI*BOTsize+krnCounterJ], gatingValTop * gatedPS_local / labelPSWSum[maxFusedIdx]);
					if (KernelGatingSwitch > -1){
						if (KernelGatingSwitch == 0) 
							atomicAdd(&kernelGating[krIdx + kernelIdx], gatedPS_local / labelPSWSum[maxFusedIdx]);
						else {
							printf("Only KernelGatingSwitch == 0 for addition is addressed.");
							assert(false);
						}
						atomicAdd(&kernelGatingHit[krIdx + kernelIdx], 1);
					}
				}
			}
		}
	}

}


// ****************************************************
// *************** Max Pooling ************************
// ****************************************************
__global__ void maxPooling(float *gatingBot, float *hiddenBot, float *gatingTop, float *policyTable,
	int kernelSize,
	int kernelLength,
	int BOTsize,
	int stride,
	int pad,
	int reduceInitSize,
	int ptLength,
	int ptOffset,
	int secondStageMode){
	assert ((kernelSize == 2) || (kernelSize == 3) || (kernelSize == 7));

	// Constant memory
	const uint32_t thdIdx = threadIdx.x;
	const uint32_t blIdx = (blockIdx.x * gridDim.y * gridDim.z) + (blockIdx.y * gridDim.z) + (blockIdx.z);
	const int32_t rfIdxy = blockIdx.y * stride - pad;
	const int32_t rfIdxz = blockIdx.z * stride - pad;
	const uint32_t y = thdIdx / kernelSize;
	const uint32_t z = modOpt(thdIdx, kernelSize);

	// Shared memory
	extern __shared__ float sharedMem[];	// dynamic shared memory
	float *gatedPS  = (float*) sharedMem;
	float *gatedIDX = (float*) &sharedMem[kernelLength];

	__shared__ float gatingValTop;			// static shared memory
	if (thdIdx == 0){
		gatingValTop = gatingTop[blIdx];
	}
	__syncthreads();

	// exit if the value of the top gating node is zero.
	if (gatingValTop == 0)
		return;

	if (!((rfIdxy+(int32_t)y<0) || (rfIdxy+(int32_t)y>=BOTsize) || (rfIdxz+(int32_t)z<0) || (rfIdxz+(int32_t)z>=BOTsize)))
		gatedPS[thdIdx] = hiddenBot[((blockIdx.x*BOTsize*BOTsize) + (uint32_t)rfIdxy*BOTsize + (uint32_t)rfIdxz)  +  y*BOTsize+z];
	else
		gatedPS[thdIdx] = 0.0f;		// here the padding value is zero using a constant strategy.
	__syncthreads();


	// ***************************************************************************
	if (secondStageMode == 0){		// Max-Pooling
		gatedIDX[thdIdx] = gatedPS[thdIdx]; 
		__syncthreads();

		// 1) reduce the max array shared memory using the half number of threads.
		for (uint32_t cnt = reduceInitSize>>1; cnt > 0; cnt >>= 1){
			if (thdIdx < cnt && thdIdx+cnt < kernelLength && gatedPS[thdIdx] < gatedPS[thdIdx+cnt])
				gatedPS[thdIdx] = gatedPS[thdIdx+cnt];
			__syncthreads();
		}
		assert (gatedPS[0] != 0);

		// 2) weight sum the gatingBot nodes using one thread to avoid multiple max selection. It might be slower but safer.
		if (thdIdx == 0)
			for (uint32_t cnt=0; cnt<kernelLength; cnt++)
				if (gatedIDX[cnt] == gatedPS[0]){
					atomicAdd(&gatingBot[((blockIdx.x*BOTsize*BOTsize) + (uint32_t)rfIdxy*BOTsize + (uint32_t)rfIdxz)  +  (cnt/kernelSize)*BOTsize+modOpt(cnt, kernelSize)], gatingValTop); // need to re-calculate y, z since this thread does not have access to the local memory scope of the max thread.
					break;
				}
	}

	// ***************************************************************************
	else if (secondStageMode == 1){	
		gatedIDX[thdIdx] = gatedPS[thdIdx]; 
		__syncthreads();

		for (uint32_t cnt = reduceInitSize>>1; cnt > 0; cnt >>= 1){
			if (thdIdx < cnt && thdIdx+cnt < kernelLength){	
				if      (gatedIDX[thdIdx]!=0.0f && gatedIDX[thdIdx+cnt]!=0.0f)
					gatedIDX[thdIdx] += gatedIDX[thdIdx+cnt];
				else if (gatedIDX[thdIdx]==0.0f && gatedIDX[thdIdx+cnt]!=0.0f)
					gatedIDX[thdIdx] = gatedIDX[thdIdx+cnt];

			}
			__syncthreads();
		}
		assert (gatedIDX[0] != 0);

		if (!((rfIdxy+(int32_t)y<0) || (rfIdxy+(int32_t)y>=BOTsize) || (rfIdxz+(int32_t)z<0) || (rfIdxz+(int32_t)z>=BOTsize)))
			atomicAdd(&gatingBot[((blockIdx.x*BOTsize*BOTsize) + (uint32_t)rfIdxy*BOTsize + (uint32_t)rfIdxz)  +  (thdIdx/kernelSize)*BOTsize+modOpt(thdIdx, kernelSize)], gatedPS[thdIdx] * gatingValTop / gatedIDX[0]);
	}

	// ***************************************************************************
	else if (secondStageMode == 2){	// 699 Selection
		__shared__ uint32_t nzCounter;
		__shared__ float mean;
		__shared__ float std;
		__shared__ uint32_t ptIndex;
		__shared__ float aMbS;
		if (thdIdx == 0){
			nzCounter = 0;
			mean = 0.0f;
			std  = 0.0f;
			ptIndex = 0;
			aMbS = 0.0f;
		}
		__syncthreads();


		// 0- Calculate The Mean of the non-zero PS by sum reduce below
		gatedIDX[thdIdx] = gatedPS[thdIdx];

		for (uint32_t cnt = reduceInitSize>>1; cnt > 0; cnt >>= 1){
			if (thdIdx < cnt && thdIdx+cnt < kernelLength){	
				gatedIDX[thdIdx] += gatedIDX[thdIdx+cnt];
			}
			__syncthreads();
		}
		if (gatedIDX[0] == 0){
			for (uint32_t counter=0; counter<kernelLength; counter++){
				printf("%f; %f; %f\n", gatedIDX[counter], gatedPS[counter], gatingValTop);
			}
		}
		assert (gatedIDX[0] != 0);	

		if (thdIdx == 0){
			mean = gatedIDX[0] / kernelLength;	
		}
		__syncthreads();


		// 1- Calculate The Standard Deviation of the non-zero PS by sum reduce below
		gatedIDX[thdIdx] = powf((gatedPS[thdIdx] - mean), 2.0f);
		__syncthreads();

		for (uint32_t cnt = reduceInitSize>>1; cnt > 0; cnt >>= 1){
			if (thdIdx < cnt && thdIdx+cnt < blockDim.x){
				gatedIDX[thdIdx] += gatedIDX[thdIdx+cnt];

			}
			__syncthreads();
		}
		if (thdIdx == 0){
			std = sqrtf(gatedIDX[0] / kernelLength);
		}
		__syncthreads();


		// 2- Search the Policy Table for the one first generates offset non-zero list of PS.
		uint32_t ptCounter;
		for (ptCounter=0; ptCounter<ptLength; ptCounter++){
			if (thdIdx==0)
				aMbS = policyTable[ptCounter*2] * mean + policyTable[ptCounter*2+1] * std;
				nzCounter = 0;
			__syncthreads();
			if (gatedPS[thdIdx] >= aMbS)
				atomicAdd(&nzCounter, 1);
			__syncthreads();
			if (nzCounter!=0)
				break;
		}
		if (thdIdx==0){
			ptIndex = min(ptLength-1, ptCounter+ptOffset);
			aMbS = policyTable[ptIndex*2] * mean + policyTable[ptIndex*2+1] * std;
		}
		__syncthreads();


		// 3- Final step, set the gatingBot according to the ptIndex
		if (gatedPS[thdIdx] >= aMbS)
			gatedIDX[thdIdx] = gatedPS[thdIdx];
		else
			gatedIDX[thdIdx] = 0.0f;
		__syncthreads();

		for (uint32_t cnt = reduceInitSize>>1; cnt > 0; cnt >>= 1){
			if (thdIdx < cnt && thdIdx+cnt < kernelLength){	
				gatedIDX[thdIdx] += gatedIDX[thdIdx+cnt];

			}
			__syncthreads();
		}
		if (gatedIDX[0] == 0){
			printf("%f; %f; %d %d %d; %d\n", gatedIDX[thdIdx], aMbS, ptIndex, ptCounter, ptOffset, nzCounter);
		}
		assert (gatedIDX[0] != 0);	


		if (gatedPS[thdIdx] >= aMbS)
			if (!((rfIdxy+(int32_t)y<0) || (rfIdxy+(int32_t)y>=BOTsize) || (rfIdxz+(int32_t)z<0) || (rfIdxz+(int32_t)z>=BOTsize)))
				atomicAdd(&gatingBot[((blockIdx.x*BOTsize*BOTsize) + (uint32_t)rfIdxy*BOTsize + (uint32_t)rfIdxz)  +  (thdIdx/kernelSize)*BOTsize+modOpt(thdIdx, kernelSize)], gatedPS[thdIdx] * gatingValTop / gatedIDX[0]); // need to re-calculate y, z since this thread does not have access to the local memory scope of the max thread.
	}
	else
		assert (false);
}




// ****************************************************
// *************** FC Linear **************************
// ****************************************************
__global__ void firstStageFC(float *theta, float *hiddenBot, float *gatingTop, float *kernel, float *abnBelief,
	float lowerBound,
	int noThreadsOrig, int noThreads,
	int noElemsOrig, int noElems,
	int dataLengthOrig, int dataLength,
	int ABNSwitch){

	const uint32_t thIdx = threadIdx.x;
	const uint32_t blIdx = blockIdx.x;
	const uint32_t blIdy = blockIdx.y;
	const uint32_t blIdz = blockIdx.z;
	const uint32_t BOTsize =gridDim.y;	
	const bool sortOrderAsc = true;

	extern __shared__ float gatedPS[];		// dynamic shared memory

	__shared__ float gatingValTop;			// static shared memory
	__shared__ float psSum;
	__shared__ float cumSum;
	__shared__ bool found;
	if (thIdx == 0){
		gatingValTop = gatingTop[blIdx*BOTsize*BOTsize + blIdy*BOTsize + blIdz];
		psSum = 0.0f;
		cumSum = 0.0f;
		found = false;
	}
	__syncthreads();

	// exit if the value of the top gating node is zero.
	if (gatingValTop == 0)
		return;

	 // 0- Initialize the gatedPS in the shared memory by multiplying the kernel with the RF
 	float localSum = 0.0f;
	for (uint32_t krnCounter = 0; krnCounter < noElems; krnCounter++){
		if ((thIdx < noThreadsOrig) && (krnCounter < noElemsOrig)){
			if (ABNSwitch==1)
				gatedPS[krnCounter*noThreads+thIdx] = abnBelief[krnCounter*noThreads+thIdx] * kernel[blIdx*dataLengthOrig + krnCounter*noThreadsOrig+thIdx] * hiddenBot[(krnCounter*noThreadsOrig+thIdx)*BOTsize*BOTsize + blIdy*BOTsize + blIdz];	
			else
				gatedPS[krnCounter*noThreads+thIdx] = kernel[blIdx*dataLengthOrig + krnCounter*noThreadsOrig+thIdx] * hiddenBot[(krnCounter*noThreadsOrig+thIdx)*BOTsize*BOTsize + blIdy*BOTsize + blIdz];
			localSum += gatedPS[krnCounter*noThreads+thIdx];
		}
		else
			gatedPS[krnCounter*noThreads+thIdx] = 0.0f;
	}
	atomicAdd(&psSum, localSum);
	__syncthreads();


	// 1- 1st Stage of the selection process
	if (psSum <= lowerBound){
		if (thIdx == 0)
			theta[blIdx] = -1000;
		return;
	}



	// BITONIC SORT: __Main loop over various subset sizes__
	for (uint32_t chunkSize = 1; chunkSize < dataLength; chunkSize <<= 1){
		for (uint32_t stride = chunkSize; stride > 0; stride >>= 1){
			for (uint32_t thCnt = thIdx; thCnt < dataLength>>1; thCnt += noThreads){
				uint32_t index  = thCnt;
				uint32_t offset = stride;
				if (stride == chunkSize){
					offset = 1 + ((index & (stride - 1)) << 1);
					index  = (index / stride) * stride  +  (stride - 1) - (index & (stride - 1));
				}
				index = (index << 1) - (index & (stride - 1));
				assert (index + offset < dataLength);

				compAnge(&gatedPS[index], &gatedPS[index + offset], sortOrderAsc);
			}
			__syncthreads();
		}
	}


	// SUM SCAN: __Main loop over fake grids to compute the sum scan of the sorted list__
	for (uint32_t gridCounter=0; gridCounter< noElems>>1; gridCounter++){
		scan_fc(&gatedPS[noThreads*2*gridCounter], cumSum, noThreads);
		__syncthreads();
	}



	// find the theta according to the lowerBound.
	for (uint32_t thCnt = thIdx; thCnt < dataLength-1; thCnt += noThreads){
		if (found)
			break;
		if ((gatedPS[thCnt] <= 0) && (gatedPS[thCnt+1] > 0)){
			theta[blIdx*BOTsize*BOTsize + blIdy*BOTsize + blIdz] = gatedPS[thCnt+1] - gatedPS[thCnt];
			found = true;
			break;
		}
	}
	__syncthreads();
	if ((thIdx==0) && !(found) && (gatedPS[dataLength-1] <= 0) && (cumSum > 0)){
		theta[blIdx*BOTsize*BOTsize + blIdy*BOTsize + blIdz] = cumSum - gatedPS[dataLength-1];
		found = true;
	}
	__syncthreads();
	if (!found)										
		for (uint32_t thCnt = thIdx; thCnt < dataLength-1; thCnt += noThreads){
			if (found)
				break;
			if ((gatedPS[thCnt] == 0) && (gatedPS[thCnt+1] > 0)){
				theta[blIdx*BOTsize*BOTsize + blIdy*BOTsize + blIdz] = gatedPS[thCnt+1];
				found = true;
				break;
			}
		}
	__syncthreads();
	if (thIdx==0){
		if (!(found) && (gatedPS[dataLength-1] == 0) && (cumSum > 0)){
			theta[blIdx*BOTsize*BOTsize + blIdy*BOTsize + blIdz] = cumSum - gatedPS[dataLength-1];
			found = true;
		}

		if (!found){
			printf("(%d, %d, %d), %f; %f \n", blockIdx.x, blockIdx.y, blockIdx.z, psSum, cumSum);
		}
		assert (found);
	}
}

__global__ void secondStageFC(float *gatingBot, float *thetaIn, float *hiddenBot, float *gatingTop, float *kernel, float *kernelGating, float *kernelGatingHit, float *policyTable, float *abnBelief,
	float epsilon,
	int noThreadsOrig,
	int noElemsOrig,
	int dataLengthOrig,
	int reduceInitSize,
	int ptLength,
	int ptOffset,
	int secondStageMode,
	int ABNSwitch,
	int KernelGatingSwitch){
	const uint32_t thIdx = threadIdx.x;
	const uint32_t blIdx = blockIdx.x;
	const uint32_t blIdy = blockIdx.y;
	const uint32_t blIdz = blockIdx.z;
	const uint32_t BOTsize =gridDim.y;

	extern __shared__ float sharedMem[];	// dynamic shared memory
	float *gatedPS  	 = (float*) sharedMem;
	float *gatedPScratch = (float*) &gatedPS[dataLengthOrig];

	__shared__ float gatingValTop;			// static shared memory
	__shared__ float theta;
	__shared__ bool labelArrayNZ;
	__shared__ uint32_t nzCounter;
	__shared__ float mean;
	__shared__ float std;
	__shared__ uint32_t ptIndex;
	__shared__ float aMbS;
	if (thIdx == 0){
		gatingValTop = gatingTop[blIdx*BOTsize*BOTsize + blIdy*BOTsize + blIdz];
		theta = thetaIn[blIdx*BOTsize*BOTsize + blIdy*BOTsize + blIdz];
		labelArrayNZ = false;
		nzCounter = 0;
		mean = 0.0f;
		std  = 0.0f;
		ptIndex = 0;
		aMbS = 0.0f;
	}
	__syncthreads();

	// exit if the value of the top gating node is zero.
	if (gatingValTop == 0 || theta <= 0)
		return;
	assert (theta > 0);


	 // 0- Initialize the gatedPS in the shared memory by multiplying the kernel with the RF
	for (uint32_t krnCounter = thIdx; krnCounter < dataLengthOrig; krnCounter+=noThreadsOrig){
		float gatedPSBUFF;
		if (ABNSwitch==1)
			gatedPSBUFF = abnBelief[krnCounter] * kernel[blIdx*dataLengthOrig + krnCounter] * hiddenBot[(krnCounter)*BOTsize*BOTsize + blIdy*BOTsize + blIdz];
		else
			gatedPSBUFF = kernel[blIdx*dataLengthOrig + krnCounter] * hiddenBot[(krnCounter)*BOTsize*BOTsize + blIdy*BOTsize + blIdz];
		if (gatedPSBUFF >= theta - epsilon){
			// printf("%f\n", gatedPSBUFF);
			gatedPS[krnCounter]	= gatedPSBUFF;
			atomicAdd(&nzCounter, 1);
			if (!labelArrayNZ)
				labelArrayNZ = true;
		}
		else
			gatedPS[krnCounter] = 0.0f;
	}
	__syncthreads();

	assert (labelArrayNZ);



	if (secondStageMode == 0){		// Top-1 Selection
		float maxFusedVal = 0.0f;
		for (uint32_t krnCounter = thIdx; krnCounter < dataLengthOrig; krnCounter+=noThreadsOrig)
			maxFusedVal = max(gatedPS[krnCounter], maxFusedVal);
		gatedPScratch[thIdx] = maxFusedVal;
		__syncthreads();

		for (uint32_t cnt = reduceInitSize>>1; cnt > 0; cnt >>= 1){
			if (thIdx < cnt && thIdx+cnt < blockDim.x && gatedPScratch[thIdx] < gatedPScratch[thIdx+cnt])
				gatedPScratch[thIdx] = gatedPScratch[thIdx+cnt];
			__syncthreads();
		}

		assert (gatedPScratch[0] != 0);	

		if (gatedPScratch[0] == maxFusedVal)
			for (uint32_t krnCounter = thIdx; krnCounter < dataLengthOrig; krnCounter+=noThreadsOrig)
				if (gatedPS[krnCounter] == maxFusedVal){
					atomicAdd(&gatingBot[(krnCounter)*BOTsize*BOTsize + blIdy*BOTsize + blIdz], gatingValTop);
					if (KernelGatingSwitch == 0) 			// Add
						atomicAdd(&kernelGating[blIdx*dataLengthOrig + krnCounter], 1);
					else
						assert(1==0);
					atomicAdd(&kernelGatingHit[blIdx*dataLengthOrig + krnCounter], 1);
				}

	}
	else if (secondStageMode == 1){	// All Selection
		// calculate the pswSum
		gatedPScratch[thIdx] = 0.0f;
		for (uint32_t krnCounter = 0; krnCounter < noElemsOrig; krnCounter++){
			float gatedPSBUFF = gatedPS[krnCounter*noThreadsOrig+thIdx];
			if (gatedPSBUFF != 0.0f){
				gatedPScratch[thIdx] += gatedPSBUFF;
			}
		}
		__syncthreads();

		for (uint32_t cnt = reduceInitSize>>1; cnt > 0; cnt >>= 1){
			if (thIdx < cnt && thIdx+cnt < blockDim.x){	
				if      (gatedPScratch[thIdx]!=0.0f && gatedPScratch[thIdx+cnt]!=0.0f)
					gatedPScratch[thIdx] += gatedPScratch[thIdx+cnt];
				else if (gatedPScratch[thIdx]==0.0f && gatedPScratch[thIdx+cnt]!=0.0f)
					gatedPScratch[thIdx] = gatedPScratch[thIdx+cnt];

			}
			__syncthreads();
		}

		if (gatedPScratch[0] == 0){
			printf("%f, %d, %f\n", theta, labelArrayNZ, gatedPScratch[thIdx]);
		}
		assert (gatedPScratch[0] != 0);	


		for (uint32_t krnCounter = thIdx; krnCounter < dataLengthOrig; krnCounter+=noThreadsOrig)
			if (gatedPS[krnCounter] != 0.0f){
				atomicAdd(&gatingBot[(krnCounter)*BOTsize*BOTsize + blIdy*BOTsize + blIdz], gatingValTop * gatedPS[krnCounter] / gatedPScratch[0]);
				if (KernelGatingSwitch == 0) 			// Add
					atomicAdd(&kernelGating[blIdx*dataLengthOrig + krnCounter], gatedPS[krnCounter] / gatedPScratch[0]);
				else
					assert(1==0);
				atomicAdd(&kernelGatingHit[blIdx*dataLengthOrig + krnCounter], 1);
			}

	}
	else if (secondStageMode == 2){	
		// 1- Calculate The Mean of the non-zero PS by sum reduce below
		gatedPScratch[thIdx] = 0.0f;
		for (uint32_t krnCounter = 0; krnCounter < noElemsOrig; krnCounter++){
			float gatedPSBUFF = gatedPS[krnCounter*noThreadsOrig+thIdx];
			if (gatedPSBUFF != 0.0f){
				gatedPScratch[thIdx] += gatedPSBUFF;
			}
		}
		__syncthreads();

		for (uint32_t cnt = reduceInitSize>>1; cnt > 0; cnt >>= 1){
			if (thIdx < cnt && thIdx+cnt < blockDim.x){	
				if      (gatedPScratch[thIdx]!=0.0f && gatedPScratch[thIdx+cnt]!=0.0f)
					gatedPScratch[thIdx] += gatedPScratch[thIdx+cnt];
				else if (gatedPScratch[thIdx]==0.0f && gatedPScratch[thIdx+cnt]!=0.0f)
					gatedPScratch[thIdx] = gatedPScratch[thIdx+cnt];

			}
			__syncthreads();
		}

		assert (gatedPScratch[0] != 0);	

		if (thIdx == 0){
			mean = gatedPScratch[0] / nzCounter;
		}
		__syncthreads();

		// 1- Calculate The Standard Deviation of the non-zero PS by sum reduce below
		gatedPScratch[thIdx] = 0.0f;
		for (uint32_t krnCounter = 0; krnCounter < noElemsOrig; krnCounter++){
			float gatedPSBUFF = gatedPS[krnCounter*noThreadsOrig+thIdx];
			if (gatedPSBUFF != 0.0f){
				gatedPScratch[thIdx] += powf((gatedPSBUFF - mean), 2.0f);
			}
		}
		__syncthreads();

		for (uint32_t cnt = reduceInitSize>>1; cnt > 0; cnt >>= 1){
			if (thIdx < cnt && thIdx+cnt < blockDim.x){
				if      (gatedPScratch[thIdx]!=0.0f && gatedPScratch[thIdx+cnt]!=0.0f)
					gatedPScratch[thIdx] += gatedPScratch[thIdx+cnt];
				else if (gatedPScratch[thIdx]==0.0f && gatedPScratch[thIdx+cnt]!=0.0f)
					gatedPScratch[thIdx] = gatedPScratch[thIdx+cnt];

			}
			__syncthreads();
		}

		if (thIdx == 0){
			std = sqrtf(gatedPScratch[0] / nzCounter);
		}
		__syncthreads();


		// 2- Search the Policy Table for the one first generates offset non-zero list of PS.
		uint32_t ptCounter;
		for (ptCounter=0; ptCounter<ptLength; ptCounter++){
			if (thIdx==0)
				aMbS = policyTable[ptCounter*2] * mean + policyTable[ptCounter*2+1] * std;
				nzCounter = 0;
			__syncthreads();
			for (uint32_t krnCounter = thIdx; krnCounter < dataLengthOrig; krnCounter+=noThreadsOrig)
				if (gatedPS[krnCounter] >= aMbS)
					atomicAdd(&nzCounter, 1);
			__syncthreads();

			if (nzCounter!=0)
				break;
		}
		if (thIdx==0){
			ptIndex = min(ptLength-1, ptCounter+ptOffset);
			aMbS = policyTable[ptIndex*2] * mean + policyTable[ptIndex*2+1] * std;
		}
		__syncthreads();

		// calculate the pswSum
		gatedPScratch[thIdx] = 0.0f;
		for (uint32_t krnCounter = 0; krnCounter < noElemsOrig; krnCounter++){
			float gatedPSBUFF = gatedPS[krnCounter*noThreadsOrig+thIdx];
			if (gatedPSBUFF >= aMbS){
				gatedPScratch[thIdx] += gatedPSBUFF;
			}
		}
		__syncthreads();

		for (uint32_t cnt = reduceInitSize>>1; cnt > 0; cnt >>= 1){
			if (thIdx < cnt && thIdx+cnt < blockDim.x){		
				if      (gatedPScratch[thIdx]!=0.0f && gatedPScratch[thIdx+cnt]!=0.0f)
					gatedPScratch[thIdx] += gatedPScratch[thIdx+cnt];
				else if (gatedPScratch[thIdx]==0.0f && gatedPScratch[thIdx+cnt]!=0.0f)
					gatedPScratch[thIdx] = gatedPScratch[thIdx+cnt];

			}
			__syncthreads();
		}

		if (gatedPScratch[0] == 0){
			printf("%f, %d, %f; %f; %d %d %d; %d\n", theta, labelArrayNZ, gatedPScratch[thIdx], aMbS, ptIndex, ptCounter, ptOffset, nzCounter);
		}
		assert (gatedPScratch[0] != 0);	


		for (uint32_t krnCounter = thIdx; krnCounter < dataLengthOrig; krnCounter+=noThreadsOrig)
			if (gatedPS[krnCounter] >= aMbS){
				atomicAdd(&gatingBot[(krnCounter)*BOTsize*BOTsize + blIdy*BOTsize + blIdz], gatingValTop * gatedPS[krnCounter] / gatedPScratch[0]);
				if (KernelGatingSwitch == 0) 			// Add
					atomicAdd(&kernelGating[blIdx*dataLengthOrig + krnCounter], gatedPS[krnCounter] / gatedPScratch[0]);
				else
					assert(1==0);
				atomicAdd(&kernelGatingHit[blIdx*dataLengthOrig + krnCounter], 1);
			}
	}
	else
		assert (false);
}


// ****************************************************
// *************** GLOBAL PORTION *********************
// ****************************************************
__global__ void convKernel(float *gatingBot, float *theta, float *hiddenBot, float *gatingTop, float *kernel, float *kernelGating, float *kernelGatingHit, float *abnBelief,
	float lowerBound,
	float epsilon,
	int noThreadsOrig, int noThreads,
	int kernelSizeOrig, int kernelSize,
	int stride,
	int pad,
	int group,
	int dataLengthSOrig, int dataLength,
	int BOTsize,
	int gridx,int gridy, int gridz,
	int sharedMemSize_1,
	int smem_1_oversized,
	int kernelLength,
	int reduceInitSize,
	int memBankRem,
	int sharedMemSize_2, int smem_2_oversized,
	int ABNSwitch,
	float EBMult,
	int KernelGatingSwitch){

	const uint32_t blIdx = blockIdx.x;
	dim3 blockDimen_1(noThreads, 1, 1);
	dim3 gridDimen(gridx, gridy, gridz);
	uint32_t topOffset = gridx * gridy * gridz;
	uint32_t botOffset = noThreadsOrig * group * BOTsize * BOTsize;
	if (smem_1_oversized == 0)
		firstStage<<<gridDimen, blockDimen_1, sharedMemSize_1>>>(&theta[(size_t)blIdx*topOffset], &hiddenBot[(size_t)blIdx*botOffset], &gatingTop[(size_t)blIdx*topOffset], kernel,
		lowerBound,
		noThreadsOrig, noThreads,
		kernelSizeOrig,
		stride,
		pad,
		group,
		dataLengthSOrig, dataLength,
		BOTsize,
		sharedMemSize_1,
		kernelLength);
	else
		firstStage_oversized<<<gridDimen, blockDimen_1, sharedMemSize_1>>>(&theta[(size_t)blIdx*topOffset], &hiddenBot[(size_t)blIdx*botOffset], &gatingTop[(size_t)blIdx*topOffset], kernel,
		lowerBound,
		noThreadsOrig, noThreads,
		kernelSizeOrig,
		stride,
		pad,
		group,
		dataLengthSOrig, dataLength,
		BOTsize,
		sharedMemSize_1,
		kernelLength);

	dim3 blockDimen_2(noThreadsOrig, 1, 1);
	secondStage<<<gridDimen, blockDimen_2, sharedMemSize_2>>>(&gatingBot[(size_t)blIdx*botOffset], &theta[(size_t)blIdx*topOffset], &hiddenBot[(size_t)blIdx*botOffset], &gatingTop[(size_t)blIdx*topOffset], kernel, kernelGating, kernelGatingHit, abnBelief, // &abnBelief[(size_t)blIdx*noThreadsOrig*group],
	epsilon,
	noThreadsOrig,
	kernelSizeOrig,
	stride,
	pad,
	group,
	BOTsize,
	kernelLength,
	smem_2_oversized,
	reduceInitSize,
	memBankRem,
	ABNSwitch,
	EBMult,
	KernelGatingSwitch);
}

__global__ void poolKernel(float *gatingBot, float *hiddenBot, float *gatingTop, float *policyTable,
	int kernelSize,
	int kernelLength,
	int BOTsize,
	int stride,
	int pad,
	int reduceInitSize,
	int ptLength,
	int ptOffset,
	int secondStageMode,
	int gridx,
	int gridy,
	int gridz,
	int sharedMemSize_1){
	const uint32_t blIdx = blockIdx.x;
	dim3 blockDimen(kernelLength, 1, 1);
	dim3 gridDimen(gridx, gridy, gridz);
	uint32_t topOffset = gridx * gridy * gridz;
	uint32_t botOffset = gridx * BOTsize * BOTsize;	
	maxPooling<<<gridDimen, blockDimen, sharedMemSize_1>>>(&gatingBot[(size_t)blIdx*botOffset],&hiddenBot[(size_t)blIdx*botOffset], &gatingTop[(size_t)blIdx*topOffset], policyTable,
	kernelSize,
	kernelLength,
	BOTsize,
	stride,
	pad,
	reduceInitSize,
	ptLength,
	ptOffset,
	secondStageMode);
}

__global__ void fcKernel(float *gatingBot, float *theta, float *hiddenBot, float *gatingTop, float *kernel, float *kernelGating, float *kernelGatingHit, float *policyTable, float *abnBelief,
	float lowerBound,
	float epsilon,
	int noThreadsOrig, int noThreads,
	int noElemsOrig, int noElems,
	int dataLengthOrig, int dataLength,
	int gridx,int gridy, int gridz,
	int sharedMemSize_1,
	int sharedMemSize_2,
	int reduceInitSize,
	int ptLength,
	int ptOffset,
	int secondStageMode,
	int ABNSwitch,
	int KernelGatingSwitch){

	const uint32_t blIdx = blockIdx.x;
	dim3 blockDimen_1(noThreads, 1, 1);
	dim3 gridDimen(gridx, gridy, gridz);
	uint32_t topOffset = gridx * gridy * gridz;
	uint32_t botOffset = dataLengthOrig * gridy * gridz;
	firstStageFC<<<gridDimen, blockDimen_1, sharedMemSize_1>>>(&theta[(size_t)blIdx*topOffset], &hiddenBot[(size_t)blIdx*botOffset], &gatingTop[(size_t)blIdx*topOffset], kernel, abnBelief, 
	lowerBound,
	noThreadsOrig, noThreads,
	noElemsOrig, noElems,
	dataLengthOrig, dataLength,
	ABNSwitch);

	dim3 blockDimen_2(noThreadsOrig, 1, 1);
	secondStageFC<<<gridDimen, blockDimen_2, sharedMemSize_2>>>(&gatingBot[(size_t)blIdx*botOffset], &theta[(size_t)blIdx*topOffset], &hiddenBot[(size_t)blIdx*botOffset], &gatingTop[(size_t)blIdx*topOffset], kernel, kernelGating, kernelGatingHit, policyTable, abnBelief, 
	epsilon,
	noThreadsOrig,
	noElemsOrig,
	dataLengthOrig,
	reduceInitSize,
	ptLength,
	ptOffset,
	secondStageMode,
	ABNSwitch,
	KernelGatingSwitch);
}
// ****************************************************
// ****************************************************
// ****************************************************

/* Function to check if x is power of 2*/
bool inline isPowerOfTwo(int x)
{
  return x && (!(x&(x-1)));
}

// ****************************************************
// *************** Entry Point is HERE ****************
// ****************************************************
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
    cudaStream_t stream, int verbose)
{
	cudaError_t err;

	// ****************************************************
	// ************** CONSTANT PORTION ********************
	// ****************************************************
	float lowerBound;
	float epsilon;

	int noThreadsOrig;	
	int noThreads;		
	int kernelSizeOrig;
	int kernelSize;		
	int dataLengthSOrig;
	int dataLength;		

	int BOTsize;		
	int stride;			
	int pad;			

	int gridx;
	int gridy;
	int gridz;
	int sharedMemSize_1;
	int smem_1_oversized = 0;
	int noElemsOrig = 0;

	int kernelLength;	
	int reduceInitSize;	
	int memBankRem;		
	int sharedMemSize_2;
	int smem_2_oversized = 0;

	int ABNSwitch;
	float EBMult;
	int KernelGatingSwitch;

	int hiddenChannel = hidden_c;
	int hiddenSize = hidden_w;
	kernelSize = k_w;	
	kernelLength = (int) pow((double) kernelSize, 2);
	stride = s_w;
	pad = p_w;
	lowerBound = 3*1e-4;
	epsilon = 2*1e-4;   

	assert(hiddenChannel % group == 0);
	noThreads = hiddenChannel / group;
	int noElems   = (int) pow((double) kernelSize, 2);

	assert(noThreads<=1024);

	noThreadsOrig = noThreads;
	kernelSizeOrig = kernelSize;

	if(!isPowerOfTwo(noThreads))
		noThreads = (int) pow(2, (double) (((int) log2((double) noThreads)) + 1));
	if(!isPowerOfTwo(noElems))
	{
		int nextPowTwo = ((int) log2((double) noElems)) + 1;
		if(nextPowTwo & 1) 
			noElems = (int) pow(2, (double) (nextPowTwo + 1));
		else
			noElems = (int) pow(2, (double) nextPowTwo);
		kernelSize = (int) sqrt(noElems);
	}
	assert(kernelSize == (int) kernelSize);

	gridx = attention_c;
	gridy = attention_h;
	gridz = attention_w;

	dataLengthSOrig = noThreads * kernelLength;
	dataLength = noThreads * noElems;
	BOTsize = hiddenSize;


	sharedMemSize_1   = noThreads * kernelLength * sizeof(float);
	if(sharedMemSize_1 > 48*1024){
		fprintf(stderr, "*** the required shared memory 1 at attentive conv layer is above the hardware limit\n");
		smem_1_oversized = 1;
		noElemsOrig = (int) ((48*1024/4 - 3*noThreads)/noThreads);
		assert(noElemsOrig < kernelSizeOrig);
		noElems = (int) pow(2, (double) (((int) log2((double) noElemsOrig)) + 1));
		dataLengthSOrig = noThreads * noElemsOrig;
		dataLength = noThreads * noElems;
		sharedMemSize_1 = noThreads * ( noElemsOrig + 2) * sizeof(float);
		if(sharedMemSize_1 > 48*1024){
			fprintf(stderr, "*** NotImplemented:\n\tthe required shared memory 1 for the second time is above the hardware limit. Reduce (kernel size / channel size) \n");
			exit(-1);
		}		
	}

	reduceInitSize = hiddenChannel / group;
	if(!isPowerOfTwo(noThreadsOrig))
		reduceInitSize = (int) pow(2, (double) (((int) log2((double) noThreadsOrig)) + 1));
	memBankRem = (k_c * k_h * k_w) % 4;
	if(memBankRem != 0)
			memBankRem = 4 - memBankRem;
	sharedMemSize_2 = (k_c * k_h * k_w * sizeof(float)) + (k_c * k_h * k_w) + noThreadsOrig * sizeof(float) + memBankRem;
	if(sharedMemSize_2 > 48*1024){
		fprintf(stderr, "*** the required shared memory 2 at attentive conv layer is above the hardware limit\n");
		smem_2_oversized = 1;
		sharedMemSize_2 = (k_c * k_h * k_w) + noThreadsOrig * sizeof(float) + memBankRem;
		if(sharedMemSize_2 > 48*1024){
			fprintf(stderr, "*** NotImplemented:\n\tthe required shared memory 2 for the second time is above the hardware limit, Reduce (kernel size / channel size)\n");
			exit(-1);
		}
	}

	ABNSwitch = 0;
	float abnBelief = 0;
	EBMult = 0.2;
	KernelGatingSwitch = 0; 

	if(verbose==1){
		printf("lowerBound: %f\nepsilon: %f\nnoThreadsOrig: %d\nnoThreads: %d\nkernelSizeOrig: %d\nkernelSize: %d\nstride: %d\npad: %d\ngroup: %d\ndataLengthSOrig: %d\ndataLength: %d\nBOTsize: %d\ngridx: %d\ngridy: %d\ngridz: %d\nsharedMemSize_1: %d\nsmem_1_oversized: %d\nnoElemsOrig: %d\nnoElems: %d\nkernelLength: %d\nreduceInitSize: %d\nmemBankRem: %d\nsharedMemSize_2: %d\nsmem_2_oversized : %d\nABNSwitch: %d\nEBMult: %f\nKernelGatingSwitch: %d\n",
		lowerBound, 
		epsilon,
		noThreadsOrig, noThreads,
		kernelSizeOrig, kernelSize,
		stride,
		pad,
		group,
		dataLengthSOrig, dataLength,
		BOTsize,
		gridx, gridy, gridz,
		sharedMemSize_1,
		smem_1_oversized,
		noElemsOrig, noElems,
		kernelLength,
		reduceInitSize,
		memBankRem,
		sharedMemSize_2,
		smem_2_oversized,
		ABNSwitch,
		EBMult,
		KernelGatingSwitch);
	}

	dim3 block(1, 1, 1);
	dim3 grid(batch_size, 1, 1);
	convKernel<<<grid, block, 0, stream>>> (output, theta, hidden, attention, weight, output_k, output_kh, &abnBelief,
		lowerBound,
		epsilon,
		noThreadsOrig, noThreads,
		kernelSizeOrig, kernelSize,
		stride,
		pad,
		group,
		dataLengthSOrig, dataLength,
		BOTsize,
		gridx, gridy, gridz,
		sharedMemSize_1,
		smem_1_oversized,
		kernelLength,
		reduceInitSize,
		memBankRem,
		sharedMemSize_2, smem_2_oversized,
		ABNSwitch,
		EBMult,
		KernelGatingSwitch);
		
    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}



void attentive_pool_cuda(
    float *hidden,
	float *attention,
	float *output,
	float *ptable,
	int ptLength, int ptOffset, int secondStageMode,
	int batch_size,
    int hidden_c, int hidden_h, int hidden_w,
	int attention_c, int attention_h, int attention_w,
    int output_c, int output_h, int output_w,
    int k_c, int k_h, int k_w,
    int s_c, int s_h, int s_w,
    int p_c, int p_h, int p_w,
    int d_c, int d_h, int d_w,
    cudaStream_t stream, int verbose)
{
	cudaError_t err;

	// ****************************************************
	// ************** CONSTANT PORTION ********************
	// ****************************************************
	int kernelSize;	

	int BOTsize;
	int stride;	
	int pad;	

	int gridx;
	int gridy;
	int gridz;
	int sharedMemSize_1;

	int kernelLength;
	int reduceInitSize;	

	int hiddenChannel = hidden_c;
	BOTsize = hidden_w;
	kernelSize = k_w;	
	kernelLength = (int) pow((double) kernelSize, 2);
	stride = s_w;
	pad = p_w;
	
	gridx = attention_c;
	gridy = attention_h;
	gridz = attention_w;

	reduceInitSize = hiddenChannel;
	if(!isPowerOfTwo(kernelLength))
		reduceInitSize = (int) pow(2, (double) (((int) log2((double) kernelLength)) + 1));

	sharedMemSize_1   = kernelLength * sizeof(float) * 2;

	assert(hiddenChannel == gridx); 
	if(sharedMemSize_1 > 48*1024){
		fprintf(stderr, "the required shared memory at attentive conv layer is above the hardware limit\n");
		exit(-1);
	}

	if(verbose==1){
		printf("kernelSize: %d\nkernelLength: %d\nBOTsize: %d\nstride: %d\npad: %d\nreduceInitSize: %d\nptLength: %d\nptOffset: %d\nsecondStageMode: %d\ngridx: %d\ngridy: %d\ngridz: %d\nsharedMemSize_1: %d\n",
		kernelSize,
		kernelLength,
		BOTsize,
		stride,
		pad,
		reduceInitSize,
		ptLength,
		ptOffset,
		secondStageMode,
		gridx,
		gridy,
		gridz,
		sharedMemSize_1);
	}
	dim3 block(1, 1, 1);
	dim3 grid(batch_size, 1, 1);
	poolKernel<<<grid, block, 0, stream>>> (output, hidden, attention, ptable,
		kernelSize,
		kernelLength,
		BOTsize,
		stride,
		pad,
		reduceInitSize,
		ptLength,
		ptOffset,
		secondStageMode,
		gridx,
		gridy,
		gridz,
		sharedMemSize_1);
		
    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void attentive_linear_cuda(
	float *hidden,
	float *weight,
	float *attention,
	float *output,
	float *theta,
	float *output_k,
	float *output_kh,
	float *ptable,
	int ptLength, int ptOffset, int secondStageMode,
	int batch_size,
	int hidden_c, int hidden_h, int hidden_w,
	int attention_c, int attention_h, int attention_w,
	int output_c, int output_h, int output_w,
	int k_c, int k_h, int k_w,
	int s_c, int s_h, int s_w,
	int p_c, int p_h, int p_w,
	int d_c, int d_h, int d_w,
	cudaStream_t stream, int verbose)
{
	cudaError_t err;

	// ****************************************************
	// ************** CONSTANT PORTION ********************
	// ****************************************************
	float lowerBound;	
	float epsilon;	

	int noThreadsOrig;
	int noThreads;		
	int noElemsOrig;	
	int noElems;		
	int dataLengthOrig;	
	int dataLength;		

	int reduceInitSize;	
	int ABNSwitch;
	int KernelGatingSwitch;

	int gridx;
	int gridy;
	int gridz;
	int sharedMemSize_1;
	int sharedMemSize_2;

	int hiddenChannel = hidden_c;
	lowerBound = 1e-5;
	epsilon = 1e-4;

	if(hiddenChannel > 1024)
		noElems = 4;
	else
		noElems = 1;

	assert((((float) hiddenChannel) / ((float) noElems)) == (int) (((float) hiddenChannel) / (float) noElems)); 
	assert(noElems < 5);
	noThreads = hiddenChannel / noElems;

	noElemsOrig = noElems;
	noThreadsOrig = noThreads;

	if(!isPowerOfTwo(noThreads))
		noThreads = (int) pow(2, (double) (((int) log2((double) noThreads)) + 1));
	if(!isPowerOfTwo(noElems))  
	{
		int nextPowTwo = ((int) log2((double) noElems)) + 1;
		if(nextPowTwo & 1) 
			noElems = (int) pow(2, (double) (nextPowTwo + 1));
		else
			noElems = (int) pow(2, (double) nextPowTwo);
	}

	dataLengthOrig = noThreadsOrig * noElemsOrig;
	dataLength = noThreads * noElems;

	assert(attention_h * attention_w == hidden_h * hidden_w); 
	assert(attention_h == attention_w); 
	gridx = attention_c;
	gridy = attention_h;
	gridz = attention_w;

	sharedMemSize_1 = noThreads * noElems * sizeof(float);
	sharedMemSize_2 = noThreadsOrig * (noElemsOrig+1) * sizeof(float);

	reduceInitSize = noThreadsOrig;
	if(!isPowerOfTwo(noThreadsOrig))
		reduceInitSize = (int) pow(2, (double) (((int) log2((double) noThreadsOrig)) + 1));

	ABNSwitch = 0;
	float abnBelief = 0;	
	KernelGatingSwitch = 0; 

	if(sharedMemSize_1 > 48*1024 or sharedMemSize_2 > 48*1024){
		fprintf(stderr, "the required shared memory at attentive conv layer is above the hardware limit\n");
		exit(-1);
	}

	if(verbose==1){
		printf("lowerBound: %f\nepsilon: %f\nnoThreadsOrig: %d\nnoThreads: %d\nnoElemsOrig: %d\nnoElems: %d\ndataLengthOrig: %d\ndataLength: %d\ngridx: %d\ngridy: %d\ngridz: %d\nsharedMemSize_1: %d\nsharedMemSize_2: %d\nreduceInitSize: %d\nptLength: %d\nptOffset: %d\nsecondStageMode: %d\nABNSwitch: %d\nKernelGatingSwitch: %d\n",
		lowerBound,
		epsilon,
		noThreadsOrig, noThreads,
		noElemsOrig, noElems,
		dataLengthOrig, dataLength,
		gridx, gridy, gridz,
		sharedMemSize_1,
		sharedMemSize_2,
		reduceInitSize,
		ptLength,
		ptOffset,
		secondStageMode,
		ABNSwitch,
		KernelGatingSwitch);
	}
	dim3 block(1, 1, 1);
	dim3 grid(batch_size, 1, 1);
	fcKernel<<<grid, block, 0, stream>>> (output, theta, hidden, attention, weight, output_k, output_kh, ptable, &abnBelief,
		lowerBound,
		epsilon,
		noThreadsOrig, noThreads,
		noElemsOrig, noElems,
		dataLengthOrig, dataLength,
		gridx, gridy, gridz,
		sharedMemSize_1,
		sharedMemSize_2,
		reduceInitSize,
		ptLength,
		ptOffset,
		secondStageMode,
		ABNSwitch,
		KernelGatingSwitch);
		
    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
