#pragma once

#include <cmath>
#include <matrix/block.h>
#include <matrix/row.h>
#include <matrix/sparse_matrix.h>
#include <equaitons_general.h>

/**
    class for a particular problem
    This class is for the advection diffusion matrix

*/


template<class Block, class Row, class Smatrix>
class advect_diff_eq2: public equaitons_general<Block, Row, Smatrix>
{
public:
      
    typedef typename Smatrix::T T;
    static const unsigned int Block_Size = Smatrix::Block_Size;
    typedef equaitons_general<Block, Row, Smatrix> eg_t;


    using eg_t::x_h;
    using eg_t::b_h;
    using eg_t::v_h;
    using eg_t::sparse_matrix_p;
    using eg_t::copy_2_CPU_arrays;
    using eg_t::copy_2_CUDA_arrays;


    advect_diff_eq2(int Nx_, int Ny_):
    Nx(Nx_), Ny(Ny_), eg_t(Nx_*Ny_)
    {
        dx = T(1)/Nx;
        dy = T(1)/Ny;
        dh = std::max(dx,dy);
    }
    ~advect_diff_eq2()
    {

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
                set_v_vector(j, k);
                set_b_vector(j, k); //first set set_x_vector vector because rhs depends on the solution for the advection diffusion scheme

            }
        }
        delete R0;
        delete Bd, Bxp, Byp, Bxm, Bym;

    }


    Block *Bd, *Bxp, *Byp, *Bxm, *Bym;
    Row* R0;
private:
    const double ccc = 0.5;
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
        //we have four corners plus four lines

        
        T b11 = T(0);
        T b12 = T(0);
        T b21 = T(0);
        T b22 = T(0);

        if(j==0)
        {
            b11 = T(1)/dt + T(ccc)/dh*(v_h[indb(j+1,k,0)] + v_h[indb(j,k,0)]) + T(1)/dh/dh/Re*(T(4)+T(1));
            b12 = T(ccc)/dh*(v_h[indb(j+1,k,1)] + v_h[indb(j,k,1)]);
        }
        else if(j==Nx-1)
        {
            b11 = T(1)/dt + T(ccc)/dh*(-v_h[indb(j,k,0)] - v_h[indb(j-1,k,0)])  + T(1)/dh/dh/Re*(T(4)+T(1));
            b12 = T(ccc)/dh*(-v_h[indb(j,k,1)] - v_h[indb(j-1,k,1)]);
        }
        else
        {
            b11 = T(1)/dt + T(ccc)/dh*(v_h[indb(j+1,k,0)] - v_h[indb(j-1,k,0)]) + T(1)/dh/dh/Re*(T(4));
            b12 = T(ccc)/dh*(v_h[indb(j+1,k,1)] - v_h[indb(j-1,k,1)]);
        }
        if(k==0)
        {
            b21 = T(ccc)/dh*(v_h[indb(j,k+1,1)] + v_h[indb(j,k,1)]);
            b22 = T(1)/dt + T(ccc)/dh*(v_h[indb(j,k+1,1)] + v_h[indb(j,k,1)]) + T(1)/dh/dh/Re*(T(4)+T(1));
        }
        else if(k==Ny-1)
        {
            b21 = T(ccc)/dh*(-v_h[indb(j,k,1)] - v_h[indb(j,k-1,1)]);
            b22 = T(1)/dt + T(ccc)/dh*(-v_h[indb(j,k,1)] - v_h[indb(j,k-1,1)]) + T(1)/dh/dh/Re*(T(4)+T(1));
        }
        else
        {   
            b21 = T(ccc)/dh*(v_h[indb(j,k+1,1)] - v_h[indb(j,k-1,1)]);
            b22 = T(1)/dt + T(ccc)/dh*(v_h[indb(j,k+1,1)] - v_h[indb(j,k-1,1)]) + T(1)/dh/dh/Re*(T(4));
        }

        Bd->set_block({b11, b12, b21, b22});

    }
    void set_xp_block(int j, int k)
    {   
        Bxp->reset_block();
        if(j<Nx-1)
        {
            Bxp->set_block({T(ccc)/dh*v_h[indb(j,k,0)] - T(1)/dh/dh/Re*(T(1)),T(0),T(0),T(ccc)/dh*v_h[indb(j,k,0)] - T(1)/dh/dh/Re*(T(1))});
        }
        else
        {
            Bd->update_set_block({-T(ccc)/dh*v_h[indb(j,k,0)],T(0),T(0),-T(ccc)/dh*v_h[indb(j,k,0)]});
        }
    }    
    void set_yp_block(int j, int k)
    {   
        Byp->reset_block();
        if(k<Ny-1)
        {
            Byp->set_block({T(ccc)/dh*v_h[indb(j,k,1)] - T(1)/dh/dh/Re*(T(1)),T(0),T(0),T(ccc)/dh*v_h[indb(j,k,1)] - T(1)/dh/dh/Re*(T(1))});
        }
        else
        {
            Bd->update_set_block({-T(ccc)/dh*v_h[indb(j,k,1)],T(0),T(0),-T(ccc)/dh*v_h[indb(j,k,1)]});
        }
    }    
    void set_xm_block(int j, int k)
    {   
        Bxm->reset_block();
        if(j>0)
        {
            Bxm->set_block({-T(ccc)/dh*v_h[indb(j,k,0)] - T(1)/dh/dh/Re*(T(1)),T(0),T(0),-T(ccc)/dh*v_h[indb(j,k,0)] - T(1)/dh/dh/Re*(T(1))});
        }
        else
        {
            Bd->update_set_block({T(ccc)/dh*v_h[indb(j,k,0)],T(0),T(0),T(ccc)/dh*v_h[indb(j,k,0)]});
        }
    }    
    void set_ym_block(int j, int k)
    {   
        Bym->reset_block();
        if(k>0)
        {
            Bym->set_block({-T(ccc)/dh*v_h[indb(j,k,1)] - T(1)/dh/dh/Re*(T(1)),T(0),T(0),-T(ccc)/dh*v_h[indb(j,k,1)] - T(1)/dh/dh/Re*(T(1))});
        }
        else
        {
            Bd->update_set_block({T(ccc)/dh*v_h[indb(j,k,1)],T(0),T(0),T(ccc)/dh*v_h[indb(j,k,1)]});
        }
    }    

    //sets vectors on host
    void set_b_vector(int j, int k)
    {
        b_h[indb(j,k,0)] = cos(M_PI*T(j)/T(Nx-1));//+x_h[indb(j,k,0)];
        b_h[indb(j,k,1)] = sin(M_PI*T(k)/T(Ny-1));//+x_h[indb(j,k,1)]; 
    }

    void set_v_vector(int j, int k)
    {
        v_h[indb(j,k,0)] = 1.0;//sin(M_PI*T(j)/T(Nx-1));
        v_h[indb(j,k,1)] = 2.0;//sin(M_PI*T(k)/T(Ny-1));
    }

    void set_x_vector(int j, int k)
    {
        x_h[indb(j,k,0)] = 0.0;//sin(M_PI*T(j)/T(Nx-1));
        x_h[indb(j,k,1)] = 0.0;//sin(M_PI*T(j)/T(Nx-1));
    }

     inline int ind(int j, int k)
    {
        return (j)*Ny+(k);
    }
    inline int indb(int j, int k, int l)
    {
        return Block_Size*(ind(j, k)) + (l);
    }
   




};