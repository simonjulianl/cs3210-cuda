/*
	CS3210 Assignment 2
	CUDA Virus Scanning

	Most of your CUDA code should go in here.

	Feel free to change any code in the skeleton, as long as you conform
	to the input and output formats specified in the assignment pdf.

	If you rename this file or add new files, remember to modify the
	Makefile! Just make sure (haha) that the default target still builds
	your program, and you don't rename the program (`scanner`).

	The skeleton demonstrates how asnychronous kernel launches can be
	done; it is up to you to decide (and implement!) the parallelisation
	paradigm for the kernel. The provided implementation is not great,
	since it launches one kernel per file+signature combination (a lot!).
	You should try to do more work per kernel in your implementation.

	You can launch as many kernels as you want; if any preprocessing is
	needed for your algorithm of choice, you can also do that on the GPU
	by running different kernels.

	'defs.h' contains the definitions of the structs containing the input
	and signature data parsed by the provided skeleton code; there should
	be no need to change it, but you can if you want to.

	'common.cpp' contains the aforementioned parsing for the input files.
	The input files are already efficiently read with mmap(), so there
	should be little to no gain trying to optimise that portion of the
	skeleton.

	Remember: print any debugging statements to STDERR!
*/

#include <vector>
#include <iomanip>
#include <iostream>

#include "defs.h"
#define NO_OF_CHARS 256
#define NUM_THREADS_PER_BLOCK 1024
#define BM_PARTITION_SIZE 1024

__device__ void bruteForce(const char* file_data, size_t file_len, const char* signature, size_t len, int* d_sig_match) {
	int index;
	for (int j = 0; j < len; j++) {
		index = j + blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= file_len) {
			return;
		} 
		
		if (signature[j] != '?' && file_data[index] != signature[j]) {
			return; 
		}

		if (j == len - 1) {
			*d_sig_match = 1;
			return;
		}
	}
}


/* A pattern searching function that uses Bad Character Heuristic of Boyer Moore Algorithm, inspired by GFG */
__device__ void boyerMoore(const char* file_data, size_t file_len, const char* signature, size_t len, int* d_sig_match, int* badchar) {
	size_t n = BM_PARTITION_SIZE;

    int s = 0;
	int startIndex = blockIdx.x * blockDim.x * BM_PARTITION_SIZE + threadIdx.x * BM_PARTITION_SIZE;
	if (startIndex >= file_len) {
		return;
	}

    while (s <= (n - len)) {
        int j = len - 1;
		
		if (startIndex + s + j >= file_len) {
			return; 
		}

        while (j >= 0 && signature[j] == file_data[startIndex + s + j]) {
			// printf("signature char: %c file char: %c\n", signature[j], file_data[startIndex + s + j]);
            j--;
        }

        if (j < 0) {
            *d_sig_match = 1;
            return;
        } else {
			int temp = (int) file_data[startIndex + s + j];
			if (temp >= 0 && temp < NO_OF_CHARS) {
				s += max(1, j - badchar[temp]);
			} else {
				s += max(1, j + 1);
			}
        }
    }
}

// __global__ void matchFile(const char* file_data, size_t file_len, const char* signature, size_t len, int* d_sig_match, int* preprocessed_data)
__global__ void matchFile(const char* file_data, size_t file_len, const char* signature, size_t len, int* d_sig_match)
{
	// TODO: your code!
	bruteForce(file_data, file_len, signature, len, d_sig_match);
	// boyerMoore(file_data, file_len, signature, len, d_sig_match, preprocessed_data);
}

__global__ void computeBadChar(const char* signature, size_t len, int badchar[NO_OF_CHARS]) {
	int i; 
	for (i = 0; i < NO_OF_CHARS; i++) {
		badchar[i] = -1;
	}

	for (i = 0; i < len; i++) {
		badchar[(int) signature[i]] = i;
	}
}

__device__ char convertDecToHex(int x) {
	if (x >= 10) { 
		return 'a' + x - 10;
	} else {
		return '0' + x; 
	}
}

__global__ void uint8_to_hex_char_array(uint8_t *v, const size_t s, char *out) {
	// size of out = s * 2
	int value, leading, remainder; 
	int current_index = blockDim.x * blockIdx.x + threadIdx.x; 
	if (current_index >= s) {
		return;
	}

	value = static_cast<int>(v[current_index]);
	leading = value >> 4; 
	remainder = value & 0xf;
	out[current_index << 1] = convertDecToHex(leading);
	out[(current_index << 1) + 1] = convertDecToHex(remainder);
}

void runScanner(std::vector<Signature>& signatures, std::vector<InputFile>& inputs)
{
	{
		cudaDeviceProp prop;
		check_cuda_error(cudaGetDeviceProperties(&prop, 0));

		fprintf(stderr, "cuda stats:\n");
		fprintf(stderr, "  # of SMs: %d\n", prop.multiProcessorCount);
		fprintf(stderr, "  global memory: %.2f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
		fprintf(stderr, "  shared mem per block: %zu bytes\n", prop.sharedMemPerBlock);
		fprintf(stderr, "  constant memory: %zu bytes\n", prop.totalConstMem);
		fprintf(stderr, "  max threads per block: %d threads\n", prop.maxThreadsPerBlock);
	}

	/*
		Here, we are creating one stream per file just for demonstration purposes;
		you should change this to fit your own algorithm and/or implementation.
	*/
	std::vector<cudaStream_t> streams {};
	streams.resize(inputs.size());

	std::vector<char*> file_bufs {};

	for(size_t i = 0; i < inputs.size(); i++)
	{
		cudaStreamCreate(&streams[i]);

		// allocate memory on the device for the uint
		uint8_t* ptr_ori = 0; 
		check_cuda_error(cudaMalloc(&ptr_ori, sizeof(uint8_t) * inputs[i].size));

		// allocate memory on the device for the file
		char* ptr = 0;
		size_t file_char_size = inputs[i].size * 2; // 1 byte = 2 hex
		check_cuda_error(cudaMalloc(&ptr, sizeof(char) * file_char_size));

		// copy the data to gpu
		cudaMemcpyAsync(
			ptr_ori, 
			inputs[i].data,
			inputs[i].size,
			cudaMemcpyHostToDevice, 
			streams[i]
		);    

		int num_blocks = (inputs[i].size + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK; 
		uint8_to_hex_char_array<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, streams[i]>>>(
			ptr_ori, 
			inputs[i].size,
			ptr
		);

		cudaFree(ptr_ori); 
		file_bufs.push_back(ptr);
	}

	for(auto& s : streams)
		cudaStreamSynchronize(s);

	// allocate memory for the signatures
	std::vector<char*> sig_bufs {};
	// std::vector<int*> preprocessed_bar_chars {};

	for(size_t i = 0; i < signatures.size(); i++)
	{
		// signature pointer on device
		char* ptr = 0;
		check_cuda_error(cudaMalloc(&ptr, signatures[i].size));
		cudaMemcpy(ptr, signatures[i].data, signatures[i].size, cudaMemcpyHostToDevice);
		sig_bufs.push_back(ptr);

		// bad char pointer on device
		// int* badchar = 0;
		// check_cuda_error(cudaMalloc(&badchar, sizeof(int) * NO_OF_CHARS));
		// preprocessed_bar_chars.push_back(badchar);
	}

	// for(size_t i = 0; i < signatures.size(); i++) {
	// 	dim3 numBlock(1);
	// 	dim3 numThread(1);
	// 	computeBadChar<<<numBlock, numThread>>>(
	// 		sig_bufs[i], 
	// 		signatures[i].size, 
	// 		preprocessed_bar_chars[i]
	// 	);
	// }

	// check_cuda_error(cudaDeviceSynchronize());

	// allocate memory for the matches
	std::vector<std::vector<int*>> match_bufs {};
	for(size_t i = 0; i < file_bufs.size(); i++) 
	{
		std::vector<int*> temp {};
		for(size_t j = 0; j < signatures.size(); j++)
		{
			int* ptr = 0;
			check_cuda_error(cudaMalloc(&ptr, sizeof(int)));
			cudaMemset(ptr, 0, sizeof(int));
			temp.push_back(ptr);
		}
		match_bufs.push_back(temp);
	}

	std::vector<std::vector<int*>> host_match {};
	for(size_t i = 0; i < file_bufs.size(); i++)
	{
		std::vector<int*> temp {};
		for(size_t j = 0; j < signatures.size(); j++)
		{
			int* ptr = (int*) malloc(sizeof(int));
			*ptr = 0;
			temp.push_back(ptr);
		}
		host_match.push_back(temp);
	}

	for(size_t file_idx = 0; file_idx < file_bufs.size(); file_idx++)
	{
		// asynchronously copy the file contents from host memory
		// (the `inputs`) to device memory (file_bufs, which we allocated above)
		int actualFileStringSize = inputs[file_idx].size * 2; // 1 byte = 2 hex

		// cudaMemcpyAsync(
		// 	file_bufs[file_idx], 
		// 	file_contents[file_idx],
		// 	actualFileStringSize,
		// 	cudaMemcpyHostToDevice, 
		// 	streams[file_idx]
		// );    // pass in the stream here to do this async

		for(size_t sig_idx = 0; sig_idx < signatures.size(); sig_idx++)
		{
			// launch the kernel!
			// your job: figure out the optimal dimensions

			/*
				This launch happens asynchronously. This means that the CUDA driver returns control
				to our code immediately, without waiting for the kernel to finish. We can then
				run another iteration of this loop to launch more kernels.

				Each operation on a given stream is serialised; in our example here, we launch
				all signatures on the same stream for a file, meaning that, in practice, we get
				a maximum of NUM_INPUTS kernels running concurrently.

				Of course, the hardware can have lower limits; on Compute Capability 8.0, at most
				128 kernels can run concurrently --- subject to resource constraints. This means
				you should *definitely* be doing more work per kernel than in our example!
			*/

			int threadsPerBlock = NUM_THREADS_PER_BLOCK;
			int numBlocks = (actualFileStringSize + threadsPerBlock - 1) / threadsPerBlock; // for brute force, replace one loop with 1 thread
			// int numBlocks = (actualFileStringSize + (NUM_THREADS_PER_BLOCK * BM_PARTITION_SIZE - 1)) / (NUM_THREADS_PER_BLOCK * BM_PARTITION_SIZE); 

			matchFile<<<numBlocks, threadsPerBlock, /* shared memory per block: */ 0, streams[file_idx]>>>(
				file_bufs[file_idx], 
				actualFileStringSize, 
				sig_bufs[sig_idx], 
				signatures[sig_idx].size,
				match_bufs[file_idx][sig_idx]
				// preprocessed_bar_chars[sig_idx]
			);

			// check_cuda_error(cudaDeviceSynchronize());

			cudaMemcpyAsync(
				host_match[file_idx][sig_idx],
				match_bufs[file_idx][sig_idx], 
				sizeof(int),
				cudaMemcpyDeviceToHost,
				streams[file_idx]
			);

			// check_cuda_error(cudaDeviceSynchronize());
		}
	}

	for(auto& s : streams)
		cudaStreamSynchronize(s);

	for(size_t file_idx = 0; file_idx < file_bufs.size(); file_idx++) {
		for(size_t sig_idx = 0; sig_idx < signatures.size(); sig_idx++) {
			// printf("current file index: %zu, signature index: %zu/%zu result: %d\n", file_idx, sig_idx, signatures.size(), *host_match[file_idx][sig_idx]);
			if (*host_match[file_idx][sig_idx] == 1) {
				printf("%s: %s\n", inputs[file_idx].name.c_str(), signatures[sig_idx].name.c_str());
			}
		}
	}

	for(auto buf: match_bufs)
		for(auto b: buf)
			cudaFree(b);

	
	for(auto buf: host_match)
		for(auto b: buf)
			free(b);

	// free the device memory, though this is not strictly necessary
	// (the CUDA driver will clean up when your program exits)
	for(auto buf : file_bufs)
		cudaFree(buf);

	for(auto buf : sig_bufs)
		cudaFree(buf);

	// clean up streams (again, not strictly necessary)
	for(auto& s : streams)
		cudaStreamDestroy(s);
}
