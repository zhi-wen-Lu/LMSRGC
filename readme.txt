To compile, you need Cmake 3.8 or greater and CUDA on the path and GCC 5.4.0.
Compile command for compressor:
	cd LMSRGC
        make
Compile command for decompressor:
	cd LMSRGDE
        make
Compress commands:
	 (compress)   ./LMSRGC R-file T-file R-T-file
	 R-file: the reference genome folder;T-file: need to be compressed genome folder; R-T-file: the compressed folder.

	 The default name of chromosomes is [chr1.fa, chr2.fa, chr3.fa, chr4.fa, chr5.fa, chr6.fa, chr7.fa, 
	 chr8.fa, chr9.fa, chr10.fa, chr11.fa, chr12.fa, chr13.fa, chr14.fa, chr15.fa, chr16.fa, chr17.fa, chr18.fa,
	 chr19.fa, chr20.fa, chr21.fa, chr22.fa, chrX.fa, chrY.fa]

Decompress commands:

	(decompress)  ./LMSRGDE R-file R-T-file Tempfile De-file V-file
	 R-file:the reference folder; R-T-file: the compression folder; Tempfile: tempporary folder to store the information of LMS;
	 V-file:need to be compressed genome folder,required(If not, please pay attention to changing the path of the corresponding file during decoding), whihc is used to verify decoding results.


note:
	The algorithm requires the GCC version to be no higher than 5.4.0, otherwise, errors will be reported during the compilation process.  
The use of the original algorithm[19] for constructing SA using multiple GPUs needs to pay attention to the communication mode between GPUs, 
cuda10.0 is used when using NV-link to link GPUs for communication, and cuda9.1 is used when using PCIe to link GPUs for communication, 
and appropriate adjustments need to be made according to different communication modes during compiling the software. 
I suggest that you first complete the compilation of the original algorithm for building SA with multiple GPUs, and then compile the algorithm we proposed. 
