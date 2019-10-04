CCCOMILER = nvcc
CCFLAGS = -g -std=c++11 -Xcompiler \"-Wl,-rpath=/media/DATA/shared/AmgX/build\"
IPROJECT = -I. -I/usr/local/cuda/include -I/media/DATA/shared/AmgX/base/include/
LPROJECT =  -ldl -L/usr/local/cuda/lib64 -lcudart -lamgxsh -L/media/DATA/shared/AmgX/build 



ser:
	${CCCOMILER} ${CCFLAGS} ${IPROJECT} test1.cpp -o test1.bin ${LPROJECT} 2>result_make.txt
