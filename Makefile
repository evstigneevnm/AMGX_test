CCCOMILER = nvcc
CCFLAGS = -g -std=c++11 -ccbin=g++-5 -Xcompiler \"-Wl,-rpath=/media/DATA/shared/AmgX/build_4.8\"
IPROJECT = -I. -I/usr/local/cuda/include -I/media/DATA/shared/AmgX/base/include/ -Isource/ 
LPROJECT =  -ldl -L/usr/local/cuda/lib64 -lcudart -lcusparse -lamgxsh -L/media/DATA/shared/AmgX/build_4.8 



der:
	${CCCOMILER} -DSCALAR_TYPE=double ${CCFLAGS} ${IPROJECT} test1.cpp -o test1.bin ${LPROJECT} 2>result_make.txt
fer:
	${CCCOMILER} -DSCALAR_TYPE=float ${CCFLAGS} ${IPROJECT} test1.cpp -o test1.bin ${LPROJECT} 2>result_make.txt
