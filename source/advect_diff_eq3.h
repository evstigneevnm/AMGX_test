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
class advect_diff_eq3: public equaitons_general<Block, Row, Smatrix>
{
public:
      
    typedef typename Smatrix::T T;
    static const unsigned int Block_Size = Smatrix::Block_Size;
    typedef equaitons_general<Block, Row, Smatrix> eg_t;


    using eg_t::x_h;
    using eg_t::b_h;
    using eg_t::sparse_matrix_p;
    using eg_t::copy_2_CPU_arrays;
    using eg_t::copy_2_CUDA_arrays;


    advect_diff_eq3(int Nx_, int Ny_, int Nz_):
    Nx(Nx_), Ny(Ny_), eg_t(Nx_*Ny_*Nz_)
    {
        dx = T(1)/Nx;
        dy = T(1)/Ny;
        dz = T(1)/Nz;
        dh = std::max(dx,dy,dz);
    }
    ~advect_diff_eq3()
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
        Bxm = new Block;
        Byp = new Block;
        Bym = new Block;
        Bzp = new Block;
        Bzm = new Block;
        R0 = new Row;

        for(int j=0;j<Nx;j++)
        { 
            for(int k=0;k<Ny;k++)
            {
                for(int l=0;l<Nz;l++)
                {

                    set_row(j, k, l);                
                    sparse_matrix_p->add_row(*R0);
                    set_x_vector(j, k, l);
                    set_b_vector(j, k, l); //first set set_x_vector vector because rhs depends on the solution for the advection diffusion scheme
                }
            }
        }
        delete R0;
        delete Bd, Bxp, Byp, Bxm, Bym, Bzp, Bzm;

    }


    Block *Bd, *Bxp, *Byp, *Bxm, *Bym, *Bzp, *Bzm;
    Row* R0;
private:
    double ccc = 0.5;
    int Nx, Ny, Nz;
    T dt, Re;
    T dx, dy, dz, dh;

    void set_row(int j, int k, int l)
    {
        
        set_d_block(j, k, l);
        set_xp_block(j, k, l);
        set_xm_block(j, k, l);
        set_yp_block(j, k, l);
        set_ym_block(j, k, l);
        set_zp_block(j, k, l);
        set_zm_block(j, k, l);
        
        R0->set_reserve_row(ind(j,k,l), 7);
        
        R0->add_block(*Bd, ind(j,k,l));
        R0->add_block(*Bxp, ind(j+1,k,l));
        R0->add_block(*Bxm, ind(j-1,k,l));
        R0->add_block(*Byp, ind(j,k+1,l));
        R0->add_block(*Bym, ind(j,k-1,l));
        R0->add_block(*Bzp, ind(j,k,l+1));
        R0->add_block(*Bzm, ind(j,k,l-1));


    }  

    void set_d_block(int j, int k, int l)
    {   
        //take boundary conditions into account!
        //we have four conors plus four lines

        
        T b11 = T(0);
        T b12 = T(0);
        T b13 = T(0);

        T b21 = T(0);
        T b22 = T(0);
        T b23 = T(0);
       
        T b31 = T(0);
        T b32 = T(0);
        T b33 = T(0);

        //ux*(ux0_x)+uy*(ux0_y)+uz*(ux0_z)+ux0*(ux_x)+uy0*(ux_y)+uz0*(ux_z)
        //  ux_x = u_{j+1}-u_{j-1}
        //
        //
        //
        //ux*(uy0_x)+uy*(uy0_y)+uz*(uy0_z)+ux0*(uy_x)+uy0*(uy_y)+uz0*(uy_z)
        //ux*(uz0_x)+uy*(uz0_y)+uz*(uz0_z)+ux0*(uz_x)+uy0*(uz_y)+uz0*(uz_z)

        if(j==0)
        {

        }
        else if(j==Nx-1)
        {

        }
        else
        {
            b11 = T(1)/dt + T(ccc)/dh*(x_h[indb(j+1,k,l,0)] - x_h[indb(j-1,k,l,0)]) + T(1)/dh/dh/Re*(T(6));
            b12 = T(ccc)/dh*(x_h[indb(j+1,k,l,1)] - x_h[indb(j-1,k,l,1)]);
            b13 = T(ccc)/dh*(x_h[indb(j+1,k,l,1)] - x_h[indb(j-1,k,l,1)]);
        }
        if(k==0)
        {
   
        }
        else if(k==Ny-1)
        {

        }
        else
        {   

        }

        Bd->set_block({b11, b12, b21, b22});

    }
    void set_xp_block(int j, int k, int j)
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
    void set_yp_block(int j, int k, int j)
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
    void set_xm_block(int j, int k, int j)
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
    void set_ym_block(int j, int k, int j)
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
    void set_b_vector(int j, int k, int l)
    {
        b_h[indb(j,k,j,0)] = sin(M_PI*T(j)/T(Nx-1))+x_h[indb(j,k,l,0)];
        b_h[indb(j,k,j,1)] = sin(M_PI*T(k)/T(Ny-1))+x_h[indb(j,k,l,1)];
        b_h[indb(j,k,j,2)] = sin(M_PI*T(l)/T(Nz-1))+x_h[indb(j,k,l,2)]; 

    }
    void set_x_vector(int j, int k, int l)
    {
        x_h[indb(j,k,l,0)] = T(0);
        x_h[indb(j,k,l,1)] = T(0);
        x_h[indb(j,k,l,2)] = T(0);
    }

    inline int ind(int j, int k, int l)
    {
        return (j)*Ny*Nz+(k)*Nz+(l);
    }
    inline int indb(int j, int k, int l, int b)
    {
        return Block_Size*(ind(j, k, l)) + (b);
    }
   




};