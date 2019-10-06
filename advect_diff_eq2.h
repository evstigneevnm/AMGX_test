#pragma once

#include <cmath>
#include <matrix/block.h>
#include <matrix/row.h>
#include <matrix/sparse_matrix.h>


template<class Block, class Row, class Smatrix>
class advect_diff_eq2
{
public:
    typedef typename Smatrix::T T;
    static const unsigned int Block_Size = Smatrix::Block_Size;

    advect_diff_eq2(int Nx_, int Ny_):
    Nx(Nx_), Ny(Ny_)
    {
        dx = T(1)/Nx;
        dy = T(1)/Ny;
        dh = std::max(dx,dy);

        b_h = (double*)malloc(Nx*Ny*Block_Size*sizeof(T));
        x_h = (double*)malloc(Nx*Ny*Block_Size*sizeof(T));
        sparse_matrix_p = new Smatrix(Nx*Ny);
        init_CUDA_arrays();

    }
    ~advect_diff_eq2()
    {
        delete sparse_matrix_p;
        free_C_array(b_h);
        free_C_array(x_h);
        free_CUDA_arrays();
    }

    void set_parameters(T dt_, T Re_)
    {
        dt = dt_;
        Re = Re_;
    }

    void copy_results()
    {
        copy_2_CPU_arrays();
    }

    void form_CUDA_arrays()
    {
        form_C_arrays();
        copy_2_CUDA_arrays();
    }

    void form_C_arrays()
    {
        Bd = new Block;
        Bxp = new Block;
        Byp = new Block;
        Bxm = new Block;
        Bym = new Block;
        R0 = new Row;

        for(int j=0;j<Nx;j++)
        { 
            for(int k=0;k<Ny;k++)
            {

                set_row(j, k);                
                sparse_matrix_p->add_row(*R0);
                set_x_vector(j, k);
                set_b_vector(j, k); //first set set_x_vector vector because rhs depends on the solution for the advection diffusion scheme


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

    void print_rows(bool force_print_ = false)
    {
        if(Nx*Ny<26)
        {
            sparse_matrix_p->print_rows();
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
    double ccc = 0.5;
    int Nx, Ny;
    T dt, Re;
    T dx, dy, dh;

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
        //take boundary conditions into account!
        //we have four conors plus four lines

        
        T b11 = T(0);
        T b12 = T(0);
        T b21 = T(0);
        T b22 = T(0);

        if(j==0)
        {
            b11 = T(1)/dt + T(ccc)/dh*(x_h[indb(j+1,k,0)] + x_h[indb(j,k,0)]) + T(1)/dh/dh/Re*(T(4)+T(1));
            b12 = T(ccc)/dh*(x_h[indb(j+1,k,1)] + x_h[indb(j,k,1)]);
        }
        else if(j==Nx-1)
        {
            b11 = T(1)/dt + T(ccc)/dh*(-x_h[indb(j,k,0)] - x_h[indb(j-1,k,0)])  + T(1)/dh/dh/Re*(T(4)+T(1));
            b12 = T(ccc)/dh*(-x_h[indb(j,k,1)] - x_h[indb(j-1,k,1)]);
        }
        else
        {
            b11 = T(1)/dt + T(ccc)/dh*(x_h[indb(j+1,k,0)] - x_h[indb(j-1,k,0)]) + T(1)/dh/dh/Re*(T(4));
            b12 = T(ccc)/dh*(x_h[indb(j+1,k,1)] - x_h[indb(j-1,k,1)]);
        }
        if(k==0)
        {
            b21 = T(ccc)/dh*(x_h[indb(j,k+1,1)] + x_h[indb(j,k,1)]);
            b22 = T(1)/dt + T(ccc)/dh*(x_h[indb(j,k+1,1)] + x_h[indb(j,k,1)]) + T(1)/dh/dh/Re*(T(4)+T(1));
        }
        else if(k==Ny-1)
        {
            b21 = T(ccc)/dh*(-x_h[indb(j,k,1)] - x_h[indb(j,k-1,1)]);
            b22 = T(1)/dt + T(ccc)/dh*(-x_h[indb(j,k,1)] - x_h[indb(j,k-1,1)]) + T(1)/dh/dh/Re*(T(4)+T(1));
        }
        else
        {   
            b21 = T(ccc)/dh*(x_h[indb(j,k+1,1)] - x_h[indb(j,k-1,1)]);
            b22 = T(1)/dt + T(ccc)/dh*(x_h[indb(j,k+1,1)] - x_h[indb(j,k-1,1)]) + T(1)/dh/dh/Re*(T(4));
        }

        Bd->set_block({b11, b12, b21, b22});

    }
    void set_xp_block(int j, int k)
    {   
        Bxp->reset_block();
        if(j<Nx-1)
        {
            Bxp->set_block({T(ccc)/dh*x_h[indb(j,k,0)] - T(1)/dh/dh/Re*(T(1)),T(0),T(0),T(ccc)/dh*x_h[indb(j,k,0)] - T(1)/dh/dh/Re*(T(1))});
        }
        else
        {
            Bd->update_set_block({-T(ccc)/dh*x_h[indb(j,k,0)],T(0),T(0),-T(ccc)/dh*x_h[indb(j,k,0)]});
        }
    }    
    void set_yp_block(int j, int k)
    {   
        Byp->reset_block();
        if(k<Ny-1)
        {
            Byp->set_block({T(ccc)/dh*x_h[indb(j,k,1)] - T(1)/dh/dh/Re*(T(1)),T(0),T(0),T(ccc)/dh*x_h[indb(j,k,1)] - T(1)/dh/dh/Re*(T(1))});
        }
        else
        {
            Bd->update_set_block({-T(ccc)/dh*x_h[indb(j,k,1)],T(0),T(0),-T(ccc)/dh*x_h[indb(j,k,1)]});
        }
    }    
    void set_xm_block(int j, int k)
    {   
        Bxm->reset_block();
        if(j>0)
        {
            Bxm->set_block({-T(ccc)/dh*x_h[indb(j,k,0)] - T(1)/dh/dh/Re*(T(1)),T(0),T(0),-T(ccc)/dh*x_h[indb(j,k,0)] - T(1)/dh/dh/Re*(T(1))});
        }
        else
        {
            Bd->update_set_block({T(ccc)/dh*x_h[indb(j,k,0)],T(0),T(0),T(ccc)/dh*x_h[indb(j,k,0)]});
        }
    }    
    void set_ym_block(int j, int k)
    {   
        Bym->reset_block();
        if(k>0)
        {
            Bym->set_block({-T(ccc)/dh*x_h[indb(j,k,1)] - T(1)/dh/dh/Re*(T(1)),T(0),T(0),-T(ccc)/dh*x_h[indb(j,k,1)] - T(1)/dh/dh/Re*(T(1))});
        }
        else
        {
            Bd->update_set_block({T(ccc)/dh*x_h[indb(j,k,1)],T(0),T(0),T(ccc)/dh*x_h[indb(j,k,1)]});
        }
    }    

    //sets vectors on host
    void set_b_vector(int j, int k)
    {
        b_h[indb(j,k,0)] = sin(M_PI*T(j)/T(Nx-1))+x_h[indb(j,k,0)];
        b_h[indb(j,k,1)] = sin(M_PI*T(k)/T(Ny-1))+x_h[indb(j,k,1)]; 
    }
    void set_x_vector(int j, int k)
    {
        x_h[indb(j,k,0)] = T(0);
        x_h[indb(j,k,1)] = T(0);
    }

    void init_CUDA_arrays()
    {
        b_d =  device_allocate<T>(Nx*Ny*Block_Size);
        x_d =  device_allocate<T>(Nx*Ny*Block_Size);
    }
    void copy_2_CUDA_arrays()
    {
        host_2_device_cpy<T>(b_d, b_h, Nx*Ny*Block_Size);
        host_2_device_cpy<T>(x_d, x_h, Nx*Ny*Block_Size);        
        sparse_matrix_p->form_matrix_gpu();
    }

    void copy_2_CPU_arrays()
    {
        //device_2_host_cpy<T>(b_h, b_d, Nx*Ny*Block_Size);
        device_2_host_cpy<T>(x_h, x_d, Nx*Ny*Block_Size);        
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
        return 2*((j)*Ny+(k)) + (l);
    }
   




};