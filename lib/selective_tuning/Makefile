# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------

# Unix commands.
PYTHON := python
NVCC_COMPILE := nvcc -c -o
NVCC_LINK := nvcc -dlink -o

RM_RF := rm -rf

# Library compilation rules.
NVCC_FLAGS := -x cu -Xcompiler -fPIC -shared -arch compute_35 -code compute_35,sm_35 -rdc true
NVCC_FLAGS_LINK := -shared -arch compute_35 -code compute_35,sm_35


# File structure.
BUILD_DIR := build
INCLUDE_DIRS := include
TORCH_FFI_BUILD := build_ffi.py
ATTENTIVE_KERNEL := $(BUILD_DIR)/attentive_cuda_kernel.o $(BUILD_DIR)/attentive_cuda_kernel_dlink.o
TORCH_FFI_TARGET := $(BUILD_DIR)/attentive_layers/_attentive_layers.so

INCLUDE_FLAGS := $(foreach d, $(INCLUDE_DIRS), -I$d)

all: $(TORCH_FFI_TARGET)

$(TORCH_FFI_TARGET): $(ATTENTIVE_KERNEL) $(TORCH_FFI_BUILD)
	$(PYTHON) $(TORCH_FFI_BUILD)
	cp -rv attentive_layers/ functions/
	rm -rf attentive_layers/

$(BUILD_DIR)/%.o: src/%.cu
	@ mkdir -p $(BUILD_DIR)
	# Separate cpp shared library that will be loaded to the extern C ffi
	$(NVCC_COMPILE) $@ $? $(NVCC_FLAGS) $(INCLUDE_FLAGS)

$(BUILD_DIR)/%_dlink.o: $(BUILD_DIR)/%.o
	# Separate nvcc dlink due to dynamic parallelism
	$(NVCC_LINK) $@ $? $(NVCC_FLAGS_LINK)


clean:
	$(RM_RF) $(BUILD_DIR) $(ATTENTIVE_KERNEL)
