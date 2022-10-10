CFLAGS = -Isrc/include -Isrc/RMQ -std=c++11

all: 
		g++ $(CFLAGS) -O0 -g -fomit-frame-pointer -fprefetch-loop-arrays -DNDEBUG  src/run.cpp src/sais.c src/libbsc/bsc.cpp src/libbsc/libbsc/adler32/adler32.cpp src/libbsc/libbsc/bwt/divsufsort/divsufsort.c src/libbsc/libbsc/bwt/bwt.cpp src/libbsc/libbsc/coder/coder.cpp src/libbsc/libbsc/coder/qlfc/qlfc.cpp src/libbsc/libbsc/coder/qlfc/qlfc_model.cpp src/libbsc/libbsc/filters/detectors.cpp src/libbsc/libbsc/filters/preprocessing.cpp src/libbsc/libbsc/libbsc/libbsc.cpp src/libbsc/libbsc/lzp/lzp.cpp src/libbsc/libbsc/platform/platform.cpp  -o  LMSRGDE 
