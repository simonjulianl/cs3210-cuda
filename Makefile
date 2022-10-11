# Makefile

NVCC          := /usr/local/cuda-11/bin/nvcc
COMMON_SRCS   := src/common.cpp
CUDA_SRCS     := src/kernel.cu

OUTPUT        := scanner

EXEC_SEQ      := scanner-seq
EXEC          := scanner
SIGNATURES    := signatures/sigs-exact.txt

SAMPLE_INPUT  := virus-0001-Win.Downloader.Banload-242+Win.Trojan.Matrix-8.in
SAMPLE_INPUT2 := virus-0002-Win.Downloader.Zlob-1779+Html.Phishing.Bank-532.in

.DEFAULT_TARGET: all

all: $(OUTPUT)

$(OUTPUT): $(CUDA_SRCS) $(COMMON_SRCS)
	$(NVCC) -std c++17 -O3 -lineinfo -dlto -arch=sm_70 -o $@ $^

run_seq:
	./${EXEC_SEQ} ${SIGNATURES} tests/${SAMPLE_INPUT}

run:
	./${EXEC} ${SIGNATURES} tests/${SAMPLE_INPUT}

run2:
	./${EXEC} ${SIGNATURES} tests/${SAMPLE_INPUT} tests/${SAMPLE_INPUT2}

clean:
	rm -f $(OUTPUT)
