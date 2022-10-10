To compile, you need Cmake 3.8 or greater and CUDA on the path and GCC 5.4.0.
Compile command:
        make
Compress and decompress commands:
	 (compress)   ./LMSRGC R-file T-file R-T-file
	 R-file: the reference genome folder;T-file: need to be compressed genome folder; R-T-file: the compressed folder.

	 The default name of chromosomes is [chr1.fa, chr2.fa, chr3.fa, chr4.fa, chr5.fa, chr6.fa, chr7.fa, 
	 chr8.fa, chr9.fa, chr10.fa, chr11.fa, chr12.fa, chr13.fa, chr14.fa, chr15.fa, chr16.fa, chr17.fa, chr18.fa,
	 chr19.fa, chr20.fa, chr21.fa, chr22.fa, chrX.fa, chrY.fa]

	(decompress)  ./LMSRGDE R-file R-T-file Tempfile De-file V-file
	 R-file:the reference folder; R-T-file: the compression folder; Tempfile: tempporary folder to store the information of LMS;
	 V-file:need to be compressed genome folder,not required,is used to verify decoding results.