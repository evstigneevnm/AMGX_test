CCCOMILER = nvcc
CCHOSTCOMPILER = gcc
AMGXINCLUDE = /media/DATA/shared/AmgX/base/include/
AMGXLIB = /media/DATA/shared/AmgX/build_4.8
CCFLAGS = -g -std=c++11 -ccbin=$(CCHOSTCOMPILER) -Xcompiler \"-Wl,-rpath=$(AMGXLIB)\"
IPROJECT = -I. -I/usr/local/cuda/include -I$(AMGXINCLUDE) -Isource/ 
LPROJECT =  -ldl -L/usr/local/cuda/lib64 -lcudart -lcusparse -lamgxsh -L$(AMGXLIB)

der:
	${CCCOMILER} -DSCALAR_TYPE=double ${CCFLAGS} ${IPROJECT} source/test1.cpp -o test1.bin ${LPROJECT} 2>result_make.txt
fer:
	${CCCOMILER} -DSCALAR_TYPE=float ${CCFLAGS} ${IPROJECT} source/test1.cpp -o test1.bin ${LPROJECT} 2>result_make.txt
