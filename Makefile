# Makefile

NVCC          := /usr/local/cuda-11/bin/nvcc
COMMON_SRCS   := src/common.cpp
CUDA_SRCS     := src/kernel.cu

OUTPUT        := scanner

EXEC_SEQ      := scanner-seq
EXEC          := scanner
SIGNATURES    := signatures/sigs-both.txt

SAMPLE_INPUT  := virus-0001-Win.Downloader.Banload-242+Win.Trojan.Matrix-8.in
SAMPLE_INPUT2 := virus-0002-Win.Downloader.Zlob-1779+Html.Phishing.Bank-532.in

.DEFAULT_TARGET: all

all: $(OUTPUT)

$(OUTPUT): $(CUDA_SRCS) $(COMMON_SRCS)
	$(NVCC) -std c++17 -O3 -lineinfo -dlto -arch=sm_70 -o $@ $^

run_seq:
	./${EXEC_SEQ} ${SIGNATURES} tests/${SAMPLE_INPUT}

run: all
	./${EXEC} ${SIGNATURES} tests/${SAMPLE_INPUT}

run_only_2: all
	./${EXEC} ${SIGNATURES} tests/${SAMPLE_INPUT2}

run_seq_2:
	./${EXEC_SEQ} ${SIGNATURES} tests/${SAMPLE_INPUT2}

run2: all
	./${EXEC} ${SIGNATURES} tests/${SAMPLE_INPUT} tests/${SAMPLE_INPUT2}

run10: all
	./scanner signatures/sigs-exact.txt tests/virus-0002-Win.Downloader.Zlob-1779+Html.Phishing.Bank-532.in tests/benign-0001.in tests/virus-0006-Win.Spyware.Banker-483.in tests/virus-0012-Win.Trojan.Bancos-1977+Html.Phishing.Auction-29.in tests/virus-0011-Win.Trojan.Sdbot-52.in tests/benign-0002.in tests/virus-0010-Win.Trojan.Corp-3.in tests/benign-0003.in tests/virus-0007-Win.Trojan.Matrix-8.in tests/virus-0001-Win.Downloader.Banload-242+Win.Trojan.Matrix-8.in

compare1: all 
	./check.py signatures/sigs-exact.txt tests/virus-0002-Win.Downloader.Zlob-1779+Html.Phishing.Bank-532.in

clean:
	rm -f $(OUTPUT)
