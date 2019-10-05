#pragma once

#include <cmath>
#include <matrix/block.h>
#include <matrix/row.h>
#include <matrix/sparse_matrix.h>


template<class Block, class Row, class Smatrix>
class advect_diff_eq
{
public:
    typedef typename Smatrix::T T;
    static const unsigned int Block_Size = Smatrix::Block_Size;

    advect_diff_eq(int Nx_, int Ny_):
    Nx(Nx_), Ny(Ny_)
    {
        b_h = (double*)malloc(Nx*Ny*Block_Size*sizeof(T));
        x_h = (double*)malloc(Nx*Ny*Block_Size*sizeof(T));
        sparse_matrix_p = new Smatrix(Nx*Ny);
        init_CUDA_arrays();

    }
    ~advect_diff_eq()
    {
        delete sparse_matrix_p;
        free_C_array(b_h);
        free_C_array(x_h);
        free_CUDA_arrays();
    }

    void form_CUDA_arrays()
    {
        form_C_arrays();
        copy_CUDA_arrays();
    }

    void form_C_arrays()
    {
        Bd = new Block();
        Bxp = new Block();
        Byp = new Block();
        Bxm = new Block();
        Bym = new Block();
        R0 = new Row();

        for(int j=0;j<Nx;j++)
        { 
            for(int k=0;k<Ny;k++)
            {

                set_row(j, k);                
                sparse_matrix_p->add_row(*R0);
                set_b_vector(j, k);
                set_x_vector(j, k);

            }
        }
        delete R0;
        delete Bd, Bxp, Byp, Bxm, Bym;

    }

    void print_matrix(bool force_print_ = false)
    {
        if(Nx*Ny<26)
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
        if(Nx*Ny<26)
        {
            std::cout << "b:" << std::endl;
            for(int j = 0; j<Nx*Ny*Block_Size;j++)
            {
                std::cout << b_h[j] << " ";
            }
            std::cout << "x:" << std::endl;
            for(int j = 0; j<Nx*Ny*Block_Size;j++)
            {
                std::cout << x_h[j] << " ";
            }            
        }
        else if(force_print_)
        {

        }

    }

    unsigned int get_number_of_nonzero_blocks()
    {
        return sparse_matrix_p->get_number_of_nonzero_blocks();
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


    


    T* b_h = nullptr;
    T* x_h = nullptr;
    T* b_d = nullptr;
    T* x_d = nullptr;
    Smatrix* sparse_matrix_p;
    Block *Bd, *Bxp, *Byp, *Bxm, *Bym;
    Row* R0;
private:
    int Nx, Ny;


    void set_row(int j, int k)
    {
        set_d_block(j, k);
        set_xp_block(j, k);
        set_yp_block(j, k);
        set_xm_block(j, k);
        set_ym_block(j, k);
        
        R0->set_reserve_row(ind(j,k), 5);
        
        R0->add_block(*Bd, ind(j,k));
        R0->add_block(*Bxp, ind(j+1,k));
        R0->add_block(*Bxm, ind(j-1,k));
        R0->add_block(*Byp, ind(j,k+1));
        R0->add_block(*Bym, ind(j,k-1));

    }  


    void set_d_block(int j, int k)
    {   
        Bd->set_block({T(10),T(-1),T(-0.2),T(10)});
    }
    void set_xp_block(int j, int k)
    {   
        if(j<Nx-1)
        {
            Bxp->set_block({T(2),T(0),T(0),T(2)});
        }
        else
        {
            Bd->update_set_block({T(10), T(0), T(0), T(10)});
        }
    }    
    void set_yp_block(int j, int k)
    {   
        if(k<Ny-1)
        {
            Byp->set_block({T(1),T(0),T(0),T(1)});
        }
        else
        {
            Bd->update_set_block({T(10), T(0), T(0), T(10)});
        }
    }    
    void set_xm_block(int j, int k)
    {   
        if(j>0)
        {
            Bxm->set_block({T(2),T(0.5),T(0),T(2)});
        }
        else
        {
            Bd->update_set_block({T(10), T(0), T(0), T(10)});
        }
    }    
    void set_ym_block(int j, int k)
    {   
        if(k>0)
        {
            Bym->set_block({T(1),T(0.5),T(1),T(1)});
        }
        else
        {
            Bd->update_set_block({T(10), T(0), T(0), T(10)});
        }
    }    

    //sets vectors on host
    void set_b_vector(int j, int k)
    {
        b_h[indb(j,k,0)] = sin(M_PI*T(j)/T(Nx-1));
        b_h[indb(j,k,1)] = sin(M_PI*T(k)/T(Ny-1)); 
    }
    void set_x_vector(int j, int k)
    {
        x_h[indb(j,k,0)] = T(0.1);
        x_h[indb(j,k,1)] = T(0.1);
    }

    void init_CUDA_arrays()
    {
        b_d =  device_allocate<T>(Nx*Ny*Block_Size);
        x_d =  device_allocate<T>(Nx*Ny*Block_Size);
    }
    void copy_CUDA_arrays()
    {
        host_2_device_cpy<T>(b_d, b_h, Nx*Ny*Block_Size);
        host_2_device_cpy<T>(x_d, x_h, Nx*Ny*Block_Size);        
        sparse_matrix_p->form_matrix_gpu();
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
    inline int ind(int j, int k)
    {
        return (j)*Ny+(k);
    }
    inline int indb(int j, int k, int l)
    {
        return 2*((j)*Ny+(k)) + l;
    }
    
};