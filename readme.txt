
//Introduction
This is a tool for compressiong FASTA format genome sequence, which is based on reference sequence and uses the Suffix Array(SA) and the longest common prefix(LCP) to find the longest matched substrings（LMS）for the compression. To speed up the operation of the algorithm, the program uses GPUs to parallelize the construction of SA, which is proposed in[1]. This package contains two programs, the LMSRGC is used for compression and the LMSRCDE is used for decompression. The compression and decompression codes are stored in two run.cpp files, which are located in the directory LMSRGC(and LMSRGCDE)/src/run.cpp. The source code for building SA using GPUs is stored in the directory "LMSRGC/src/multi-gpu-suffix-array/src/".
 
//Server Environment
The program is written in C++ and tested on Red Hat Enterprise Linux 7.9(64-bit), and with 2 RTX6000 GPUs with 24GB of RAM, and 2 * 2.6 GHz Intel Xeon Gold 6240 CPUs (18 cores) with 256GB RAM. This program requires Cmake 3.8 or greater but the version of GCC is no higher than 5.4.0. Meanwhile, this program also requires CUDA 10.0 (or 9.1) is installed in the root directory of the server.

Notice:
	The use of the original algorithm[1] for constructing SA using multiple GPUs needs to pay attention to the communication mode between GPUs, cuda10.0 is used when using NV-link to link GPUs for communication, and cuda9.1 is used when using PCIe to link GPUs for communication, and appropriate adjustments need to be made according to different communication modes during compiling the software. 

//Compile
During the compilation process, first locate the "Makefile" file, which is stored in the directory LMSRGC (or LMSRGCDE)/ , and then use the command "make" to compile the source codes.
Compile command for compressor:
	cd LMSRGC
       	make
Compile command for decompressor:
	cd LMSRGCDE
       	make

//Compress and decompress commands
Compress commands:
	(compress)   ./LMSRGC R-file T-file R-T-file
R-file is the reference genome;T-file is the target genome; R-T-file is the compressed result.
Example:    (./LMSRGC  hg17  YH  hg17-YH)
 hg17 is the reference genome; YH is the target genome; hg17-YH is the compressed result.
 The default name of chromosomes is [chr1.fa, chr2.fa, chr3.fa, chr4.fa, chr5.fa, chr6.fa, chr7.fa, chr8.fa, chr9.fa, chr10.fa, chr11.fa, chr12.fa, chr13.fa, chr14.fa, chr15.fa, chr16.fa, chr17.fa, chr18.fa, chr19.fa, chr20.fa, chr21.fa, chr22.fa, chrX.fa, chrY.fa]

Decompress commands:
(decompress)  ./LMSRGDE R-file R-T-file Tempfile De-file  V-file
R-file is the reference genome; R-T-file is the compression result; Tempfile is the tempporary folder to store the decoded information of the compression result; De-file is the decoded file; V-file is the target genome, which is required to verify decoded result. (If not required, please note changing the path of the corresponding file during the decoding process)
Example:     (./LMSRGCDE   hg17   hg17-YH  Tempfile  Decode  YH)
hg17 is the reference genome; hg17-YH is the compression result; Tempfile is the folder used to store the decoded information of the hg17-YH; Decode is  the folder used to store the decoded result; YH is the target genome for verifying the result;
The default name of the chromosome is the same as those used for compression.
//Reference
1.Florian Büren, Daniel Jünger,  Kobus, R. ,  Hundt, C. , &  Schmidt, B. . (2019). Suffix Array Construction on Multi-GPU Systems. the 28th International Symposium.
2.	Chris-Andre, L. , &  Burkhard, M. . (2014). Kmacs: the k-mismatch average common substring approach to alignment-free sequence comparison. Bioinformatics(14), 2000-8.
