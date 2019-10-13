#pragma once

#include <cstring>
#include <cmath>
#include <utils/cuda_support.h>
#include <utils/cuda_safe_call.h>
#include <cusparse.h>
#include <utils/cusparse_safe_call.h>

namespace numerical_algos
{


template<class T>
cusparseStatus_t cusparseXbsrmv(cusparseHandle_t handle_, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nb, int nnzb, const T *alpha, const cusparseMatDescr_t descr, const T *bsrVal, const int *bsrRowPtr, const int *bsrColInd, int blockDim, const T *x, const T *beta, T *y)
{
}

template<>
cusparseStatus_t cusparseXbsrmv<float>(cusparseHandle_t handle_, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nb, int nnzb, const float *alpha, const cusparseMatDescr_t descr, const float *bsrVal, const int *bsrRowPtr, const int *bsrColInd, int blockDim, const float *x, const float *beta, float *y)
{
    return cusparseSbsrmv(handle_, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y);
}

template<>
cusparseStatus_t cusparseXbsrmv<double>(cusparseHandle_t handle_, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nb, int nnzb, const double *alpha, const cusparseMatDescr_t descr, const double *bsrVal, const int *bsrRowPtr, const int *bsrColInd, int blockDim, const double *x, const double *beta, double *y)
{
    return cusparseDbsrmv(handle_, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y);
}


template<class Row>
class sparse_matrix
{
public:
    typedef typename Row::T T;
    static const unsigned int Block_Size = Row::Block_Size;

// external CPU containers for CSR format
    T *data = nullptr;
    int *IA = nullptr;
    int *JA = nullptr;
// ends
// external CPU containers for CSR format
    T *data_d = nullptr;
    int *IA_d = nullptr;
    int *JA_d = nullptr;
// ends

    sparse_matrix(int number_of_rows_): number_of_rows(number_of_rows_), nnz(0), number_of_blocks(0) {}
    ~sparse_matrix()
    {

        if(data!=nullptr)
        {
            cudaFree(data_d);
        }
        if(IA!=nullptr)
        {
            cudaFree(IA_d);
        }
        if(JA!=nullptr)
        {
            cudaFree(JA_d);
        }

        if(data!=nullptr)
        {
            free(data);
        }
        if(IA!=nullptr)
        {
            free(IA);
        }
        if(JA!=nullptr)
        {
            free(JA);
        }
        if(handle != 0)
        {
            CUSPARSE_SAFE_CALL( cusparseDestroy(handle) );
        }
    }



    void add_row(Row& row_l)
    {

        auto search = map_container.find(row_l.get_row_number());
        if(search == map_container.end())
        {
            map_container.insert({row_l.get_row_number(), row_l});
            //number_of_rows++;
            number_of_blocks+=row_l.get_number_of_nonzero_blocks();
            nnz+=row_l.get_number_of_nonzero_blocks()*Block_Size*Block_Size;
        }
        else
        {
            //TODO: generate a error
            std::cerr << "Row with number " <<  row_l.get_row_number() << " already exists!" << std::endl;
        }
    }
    
    void form_matrix()
    {
        //TODO add exception try{} catch{} blocks
        //int number_of_actual_rows = map_container.size();

        data = (T*)realloc((T*)data, nnz*sizeof(T));
        IA = (int*)realloc((int*)IA, (number_of_rows+1)*sizeof(int));
        JA = (int*)realloc((int*)JA, (number_of_blocks)*sizeof(int));
        
        int J = 0;
        IA[0]=0;

        int cummulative_blocks = 0;
        for(int j=0;j<number_of_rows;j++)
        {
            cummulative_blocks+=map_container[j].get_number_of_nonzero_blocks(); //invokes a constructor for a void row!
            IA[j+1]=cummulative_blocks;
        }
        for(auto &x: map_container)
        {
            for(auto &y: x.second.data)
            {
                JA[J] = y.first; //std::memcpy?!?
                //void* memcpy( void* dest, const void* src, std::size_t count );
                std::memcpy((T*)&data[Block_Size*Block_Size*(J++)], (const T*)y.second.data, sizeof(T)*Block_Size*Block_Size);
            }
        }
    }
    void form_matrix_gpu()
    {
        
        if(handle == 0)
            CUSPARSE_SAFE_CALL( cusparseCreate(&handle) );

        form_matrix();
        //GPU manipulaitons require inclusion of some files from <utils/>
        if(data_d == nullptr)
        {
            data_d = device_allocate<T>( nnz );
        }
        if(IA_d == nullptr)
        { 
            IA_d = device_allocate<int>( (number_of_rows+1) );
        }
        if(JA_d == nullptr)
        {
            JA_d = device_allocate<int>( (number_of_blocks) );
        }

        host_2_device_cpy<T>( data_d, data, nnz );
        host_2_device_cpy<int>( IA_d, IA, (number_of_rows+1) );
        host_2_device_cpy<int>( JA_d, JA, (number_of_blocks) );

        

    }

    void print_rows()
    {
        for(auto &x: map_container)
        {
            x.second.print_row();
        }
    }
    void print_matrix()
    {
        for(int j=0;j<number_of_rows+1;j++)
        {
            std::cout << IA[j] << " ";  
        }
        std::cout << std::endl;
        for(int j=0;j<number_of_blocks;j++)
        {
            std::cout << JA[j] << " ";  
        }
        std::cout << std::endl;
        for(int j=0;j<nnz;j++)
        {
            std::cout << data[j] << " ";  
        }
        std::cout << std::endl;        
    }
    unsigned int get_number_of_nonzero_blocks()
    {
        return number_of_blocks;
    }
    unsigned int get_block_size()
    {
        return Block_Size;
    }



    void axpy_gpu(const T* alpha, const T*& x, const T* beta, T*& y)const
    {

        cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN; 
        cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
        //T one(1);
        //T zero(0);

        cusparseMatDescr_t descr = 0; 

        CUSPARSE_SAFE_CALL( cusparseCreateMatDescr(&descr) );
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);

        //TODO how about fill rhs with zeros or something?
        //CUDA_SAFE_CALL( cudaMemcpy( x_new.d, rhs.d, rhs.N*rhs.block_sz*sizeof(T), cudaMemcpyDeviceToDevice ) );

        //TODO check whether x.block_size(), f.block_size(), block_row_size_ 
        //and block_col_size_ are the same

        //Matrix multiplication
        CUSPARSE_SAFE_CALL( cusparseXbsrmv<T>(handle, dir, trans, 
                                              number_of_rows, number_of_rows, number_of_blocks, 
                                              alpha, descr, data_d, IA_d, JA_d, 
                                              Block_Size, x, beta, y) );
    }


    //THIS IS BAD! JUST FOR TESTING PURPOSES!!!
    T residual_gpu(const T*& x_l, const T*& b_l) const
    {
        if(resid == nullptr)
        {
            resid = (T*)malloc(number_of_rows*Block_Size*sizeof(T));

        }
        if(resid_d == nullptr)
        {
            resid_d = device_allocate<T>( number_of_rows*Block_Size );
        }

        device_2_device_cpy<T>(resid_d, (T*)b_l, number_of_rows*Block_Size);
        T alpha(T(1));
        T beta(T(-1));
        axpy_gpu(&alpha, x_l, &beta, (T*&)resid_d);
        device_2_host_cpy<T>(resid, resid_d, number_of_rows*Block_Size );
        T resid_val = T(0);
        for(int j=0;j<number_of_rows*Block_Size;j++)
            resid_val+=resid[j]*resid[j];

        if(resid != nullptr)
        {
            free(resid);
        }
        if(resid_d != nullptr)
        {
            cudaFree(resid_d);
        }


        return(sqrt(resid_val));
    }


private:
    cusparseHandle_t handle=0;

    int number_of_rows;
    int number_of_blocks;
    int nnz;
    std::map<int, Row > map_container;

    mutable T* resid_d = nullptr;
    mutable T* resid = nullptr;
    
};

}