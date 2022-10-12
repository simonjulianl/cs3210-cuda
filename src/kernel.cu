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

#include "defs.h"

// Sanity check to run on CPU
/*
void cpuMatchFile(const uint8_t* file_data, size_t file_len, const uint8_t* signature, size_t len) {
	for (int i = 0; i < file_len - len + 1; i++) {
		for (int j = 0; j < len; j++) {
			if (file_data[i + j] != signature[j]) {
				break;
			}
			printf("%d %d %d\n", i + j, j, (int) file_data[i + j + 1]);
			if (j == len - 1) {
				printf("HEHEHE");
				return;
			}
		}
	}
}
*/

__global__ void matchFile(const uint8_t* file_data, size_t file_len, const uint8_t* signature, size_t len, int* d_sig_match)
{
	// TODO: your code!
	for (int j = 0; j < len; j++) {
		if (file_data[j + blockIdx.x * blockDim.x + threadIdx.x] != signature[j]) {
			break;
		}
		if (j == len - 1) {
			*d_sig_match = 1;
			return;
		}
	}
}

// Inspired by https://stackoverflow.com/questions/17261798/converting-a-hex-string-to-a-byte-array
int hex2dec(char char1) {
  if (char1 >= '0' && char1 <= '9')
    return char1 - '0';
  if (char1 >= 'A' && char1 <= 'F')
    return char1 - 'A' + 10;
  if (char1 >= 'a' && char1 <= 'f')
    return char1 - 'a' + 10;
	return 0;
}

int hex2decHelper(char char1, char char2) {
	return 16 * hex2dec(char1) + hex2dec(char2);
}

uint8_t* convertStringToByte(char* signature, size_t size) {
	uint8_t* result = (uint8_t*) malloc(sizeof(uint8_t) * (size / 2));
	for (int i = 0; i < size / 2; i ++) {
		result[i] = (uint8_t) hex2decHelper(signature[2 * i], signature[2 * i + 1]);
	}
	return result;
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
	}

	/*
		Here, we are creating one stream per file just for demonstration purposes;
		you should change this to fit your own algorithm and/or implementation.
	*/
	std::vector<cudaStream_t> streams {};
	streams.resize(inputs.size());

	std::vector<uint8_t*> file_bufs {};
	for(size_t i = 0; i < inputs.size(); i++)
	{
		cudaStreamCreate(&streams[i]);

		// allocate memory on the device for the file
		uint8_t* ptr = 0;
		check_cuda_error(cudaMalloc(&ptr, inputs[i].size));
		file_bufs.push_back(ptr);
	}

	// allocate memory for the signatures
	std::vector<uint8_t*> sig_bufs {};
	for(size_t i = 0; i < signatures.size(); i++)
	{
		uint8_t* ptr = 0;
		check_cuda_error(cudaMalloc(&ptr, signatures[i].size));
		uint8_t* signature_byte = convertStringToByte(signatures[i].data, signatures[i].size);
		cudaMemcpy(ptr, signature_byte, signatures[i].size, cudaMemcpyHostToDevice);
		sig_bufs.push_back(ptr);
	}

	// allocate memory for the matches
	std::vector<int*> match_bufs {};
	for(size_t i = 0; i < signatures.size(); i++)
	{
		int* ptr = 0;
		check_cuda_error(cudaMalloc(&ptr, sizeof(int)));
		cudaMemcpy(ptr, 0, sizeof(int), cudaMemcpyHostToDevice);
		match_bufs.push_back(ptr);
	}

	std::vector<int*> host_match {};
	for(size_t i = 0; i < signatures.size(); i++)
	{
		int* ptr = (int*) malloc(sizeof(int));
		*ptr = 0;
		host_match.push_back(ptr);
	}

	for(size_t file_idx = 0; file_idx < inputs.size(); file_idx++)
	{
		// asynchronously copy the file contents from host memory
		// (the `inputs`) to device memory (file_bufs, which we allocated above)
		cudaMemcpyAsync(file_bufs[file_idx], inputs[file_idx].data, inputs[file_idx].size,
			cudaMemcpyHostToDevice, streams[file_idx]);    // pass in the stream here to do this async

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

			// Sanity check can be useful
			/*
			if (sig_idx == 68) {
				// for (int i = 0; i < inputs[sig_idx].size; i++) {
				// 	printf("%d ", (int) inputs[file_idx].data[i]);
				// }
				printf("%s\n", signatures[sig_idx].name.c_str());

				uint8_t* temp = convertStringToByte(signatures[sig_idx].data, signatures[sig_idx].size);
				for (int i = 0; i < signatures[sig_idx].size / 2; i++) {
					printf("%d ", (int) temp[i]);
				}

				cpuMatchFile(inputs[file_idx].data, inputs[file_idx].size, temp, signatures[sig_idx].size / 2);
			}
			*/

			int threadsPerBlock = 1024;
			int blocksPerGrid = (inputs[file_idx].size + threadsPerBlock - 1) / threadsPerBlock;
			matchFile<<<blocksPerGrid, threadsPerBlock, /* shared memory per block: */ 0, streams[file_idx]>>>(
				file_bufs[file_idx], inputs[file_idx].size,
				sig_bufs[sig_idx], signatures[sig_idx].size / 2, match_bufs[sig_idx]);
			cudaDeviceSynchronize();

			cudaMemcpy(host_match[sig_idx], match_bufs[sig_idx], sizeof(int), cudaMemcpyDeviceToHost);

			if (*host_match[sig_idx] == 1) {
				printf("%s: %s\n", inputs[file_idx].name.c_str(), signatures[sig_idx].name.c_str());
			}
		}
	}

	for(auto buf: match_bufs)
		cudaFree(buf);

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
