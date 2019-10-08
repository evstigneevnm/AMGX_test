#pragma once

#include <cmath>
#include <matrix/block.h>
#include <matrix/row.h>
#include <matrix/sparse_matrix.h>

//

template<class Block, class Row, class Smatrix>
struct equaitons_general
{
    typedef typename Smatrix::T T;
    static const unsigned int Block_Size = Smatrix::Block_Size;

    //all public data location:
    T* b_h = nullptr;
    T* x_h = nullptr;
    T* b_d = nullptr;
    T* x_d = nullptr;
    Smatrix* sparse_matrix_p;
    size_t whole_size;

    //methods:
    equaitons_general(size_t whole_size_):
    whole_size(whole_size_)
    {
        
        b_h = (double*)malloc(whole_size*Block_Size*sizeof(T));
        x_h = (double*)malloc(whole_size*Block_Size*sizeof(T));
        sparse_matrix_p = new Smatrix(whole_size);
        init_CUDA_arrays();

    }
    ~equaitons_general()
    {
        delete sparse_matrix_p;
        free_C_array(b_h);
        free_C_array(x_h);
        free_CUDA_arrays();
    }



    unsigned int get_number_of_nonzero_blocks()
    {
        return sparse_matrix_p->get_number_of_nonzero_blocks();
    }

   void init_CUDA_arrays()
    {
        b_d =  device_allocate<T>(whole_size*Block_Size);
        x_d =  device_allocate<T>(whole_size*Block_Size);
    }
    void copy_2_CUDA_arrays()
    {
        host_2_device_cpy<T>(b_d, b_h, whole_size*Block_Size);
        host_2_device_cpy<T>(x_d, x_h, whole_size*Block_Size);        
        sparse_matrix_p->form_matrix_gpu();
    }

    void copy_2_CPU_arrays()
    {
        //device_2_host_cpy<T>(b_h, b_d, Nx*Ny*Block_Size);
        device_2_host_cpy<T>(x_h, x_d, whole_size*Block_Size);        
    }

    void free_CUDA_arrays()
    {
        if(x_d != nullptr)
        {
            cudaFree(x_d);
            x_d = nullptr;
        }
        if(b_d != nullptr)
        {
            cudaFree(b_d);
            b_d = nullptr;
        }
    }
    void free_C_array(T*& array)
    {
        if(array!=nullptr)
        {
            free(array);    
            array = nullptr;
        }
    }


    int *get_matrix_JA()
    {
        return sparse_matrix_p->JA;
    }
    int *get_matrix_IA()
    {
        return sparse_matrix_p->IA;
    }
    T *get_matrix_data()
    {
        return sparse_matrix_p->data;
    }
    T* get_b()
    {
        return b_h;
    }
    T* get_x()
    {
        return x_h;
    }


    int *get_matrix_CUDA_JA()
    {
        return sparse_matrix_p->JA_d;
    }
    int *get_matrix_CUDA_IA()
    {
        return sparse_matrix_p->IA_d;
    }
    T *get_matrix_CUDA_data()
    {
        return sparse_matrix_p->data_d;
    }
    T* get_b_CUDA()
    {
        return b_d;
    }
    T* get_x_CUDA()
    {
        return x_d;
    }

    void print_matrix(bool force_print_ = false)
    {
        if(whole_size<26)
        {
            sparse_matrix_p->print_matrix();
        }
        else if(force_print_)
        {
            sparse_matrix_p->print_matrix();
        }
    }

    void print_system(bool force_print_ = false)
    {
        print_matrix(force_print_);
        if(whole_size<26)
        {
            std::cout << "b:" << std::endl;
            for(int j = 0; j<whole_size*Block_Size;j++)
            {
                std::cout << b_h[j] << " ";
            }
            std::cout << "x:" << std::endl;
            for(int j = 0; j<whole_size*Block_Size;j++)
            {
                std::cout << x_h[j] << " ";
            }            
        }
        else if(force_print_)
        {

        }

    }

    void print_rows(bool force_print_ = false)
    {
        if(whole_size<26)
        {
            sparse_matrix_p->print_rows();
        }
        else if(force_print_)
        {

        }
    }




};