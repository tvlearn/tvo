# 
#
#

ifdef PARALLEL_RUNS
    OUT_DIR := out-parallel
else
    OUT_DIR := out
endif

MAKEFILE_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
REPO_DIR := $(shell realpath $(MAKEFILE_DIR)/../../)
TASK_SCRIPT_PATH := $(realpath $(TASK_SCRIPT))
TASK_SCRIPT_DIR := $(shell dirname $(realpath $(TASK_SCRIPT)))
TASK_FILENAME := $(shell basename $(TASK_SCRIPT))
OUT_PATH := $(subst .sh,,$(TASK_SCRIPT_PATH))
OUT_PATH := $(subst $(shell basename $(MAKEFILE_DIR)),$(OUT_DIR),$(OUT_PATH))
TASK_NAME := $(subst .sh,,$(TASK_FILENAME))

TASK_TARGET := $(OUT_PATH)/done.log

SLURM_FOUND := $(shell command -v sbatch 1>/dev/null && echo 1 || echo 0)


.PHONY: info
info:
	@echo "MAKEFILE_DIR      = $(MAKEFILE_DIR)"
	@echo "REPO_DIR          = $(REPO_DIR)"
	@echo "TASK_SCRIPT_PATH  = $(TASK_SCRIPT_PATH)"
	@echo "TASK_SCRIPT_DIR   = $(TASK_SCRIPT_DIR)"
	@echo "TASK_FILENAME     = $(TASK_FILENAME)"
	@echo "TASK_NAME         = $(TASK_NAME)"
	@echo "OUT_PATH          = $(OUT_PATH)"
	@echo "TASK_TARGET       = $(TASK_TARGET)"
	@echo "SLURM_FOUND       = $(SLURM_FOUND)"
	

$(TASK_TARGET):
	mkdir -p $(OUT_PATH)
ifeq ($(SLURM_FOUND),0)
	OUT_PATH=$(OUT_PATH) TASK_NAME=$(TASK_NAME) MAKEFILE_DIR=$(MAKEFILE_DIR) TASK_SCRIPT_PATH=$(TASK_SCRIPT_PATH) REPO_DIR=$(REPO_DIR) TVO_GPU=$(TVO_GPU) \
	$(MAKEFILE_DIR)/.job.sh
else
	cd $(OUT_PATH) && \
	sbatch --export=OUT_PATH=$(OUT_PATH),TASK_NAME=$(TASK_NAME),MAKEFILE_DIR=$(MAKEFILE_DIR),TASK_SCRIPT_PATH=$(TASK_SCRIPT_PATH),REPO_DIR=$(REPO_DIR),PARALLEL_RUNS=$(PARALLEL_RUNS),TVO_GPU=$(TVO_GPU) \
	$(MAKEFILE_DIR)/.job.sh 
endif


.PHONY: all
all: $(TASK_TARGET)


.PHONY: clean
clean:
	rm -rf "$(OUT_PATH)"
