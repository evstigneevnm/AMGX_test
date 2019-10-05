#include <iostream>
#include <cuda_runtime.h>
#include <amgx_c.h>
#include <utils/cuda_safe_call.h>
#include <utils/cuda_support.h>
#include <vector>
#include <map>
#include <cmath>

#include <matrix/block.h>
#include <matrix/row.h>
#include <matrix/sparse_matrix.h>

void check_memory(std::string message)
{

    size_t free_mem, total_mem;

    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    std::cout << "press Enter";
    std::getchar();
    std::cout <<  "------>"  << message << ":";
    CUDA_SAFE_CALL( cudaMemGetInfo(&free_mem, &total_mem) );
    std::cout << "total_mem:" << total_mem/1024/1024 << " free mem:" << free_mem/1024/1024  << " used:" << (total_mem - free_mem)/1024/1024 << std::endl;

}

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
        Bd->set_block({T(8),T(-1),T(-0.2),T(8)});
    }
    void set_xp_block(int j, int k)
    {   
        if(j<Nx-1)
        {
            Bxp->set_block({T(2),T(0),T(0),T(2)});
        }
        else
        {
            Bd->update_set_block({T(1), T(1), T(1), T(1)});
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
            Bd->update_set_block({T(1), T(1), T(1), T(1)});
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
            Bd->update_set_block({T(1), T(1), T(1), T(1)});
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
            Bd->update_set_block({T(1), T(1), T(1), T(1)});
        }
    }    

    //sets vectors on host
    void set_b_vector(int j, int k)
    {
        b_h[indb(j,k,0)] = sin(2*M_PI*T(j)/T(Nx));
        b_h[indb(j,k,1)] = sin(2*M_PI*T(k)/T(Ny)); 
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



template<class TTT>
void set_amgx_mode(AMGX_Mode& mode_x_l)
{
}

template<>
void set_amgx_mode<double>(AMGX_Mode& mode_x_l)
{
    mode_x_l = AMGX_mode_dDDI;
}
template<>
void set_amgx_mode<float>(AMGX_Mode& mode_x_l)
{
    mode_x_l = AMGX_mode_dFFI;
}


int main(int argc, char const *argv[])
{
    

    if(argc!=3)
    {
        printf("Usage: %s matrix_size path_to_amgx_config_file \n",argv[0]);
        return 0;
    }

    AMGX_Mode mode_x;
    AMGX_config_handle cfg_x;
    AMGX_resources_handle resources_x;
    AMGX_matrix_handle A_x;
    AMGX_vector_handle b_x, x_x;
    AMGX_solver_handle solver_x;
    AMGX_SOLVE_STATUS status_x;

    int Nx = 5, Ny = 5;
    Nx = atoi(argv[1]);
    Ny = Nx;

    if(init_cuda(10)==-1)
    {
        std::cout << "error in GPU selection." << std::endl; 
        return 0;
    }

  

    //typedefs
    const int block_size = 2;
    typedef block<block_size, double> block_t;
    typedef row<block_t> row_t;
    typedef sparse_matrix<row_t> sparse_matrix_t;
    typedef advect_diff_eq <block_t, row_t, sparse_matrix_t> adv_eq_t;

    check_memory("init");
    
    adv_eq_t ad_eq_class(Nx, Ny);

    ad_eq_class.form_CUDA_arrays();
    ad_eq_class.print_matrix();

    printf("nnzb = %i\n",ad_eq_class.get_number_of_nonzero_blocks());
    
    check_memory("adveciton_diffusion_class");
  
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    AMGX_SAFE_CALL(AMGX_install_signal_handler());
    set_amgx_mode<double>(mode_x);

    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg_x, argv[2]));
    AMGX_SAFE_CALL(AMGX_resources_create_simple(&resources_x, cfg_x));

    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg_x, "exception_handling=1"));
    check_memory("AMGX INIT");    
    AMGX_matrix_create(&A_x, resources_x, mode_x);
    AMGX_vector_create(&x_x, resources_x, mode_x);
    AMGX_vector_create(&b_x, resources_x, mode_x);
    AMGX_solver_create(&solver_x, resources_x, mode_x, cfg_x);
    check_memory("Arrays created");   


    // //test!
    // cudaFree(SMatrix.data_d);
    // SMatrix.data_d = nullptr;

//AMGX_RC AMGX_API AMGX_matrix_upload_all(AMGX_matrix_handle mtx, int n,
//                                          int nnz, int block_dimx, int block_dimy, const int *row_ptrs,
//                                          const int *col_indices, const void *data, const void *diag_data);
    
    AMGX_matrix_upload_all(A_x, Nx*Ny, ad_eq_class.get_number_of_nonzero_blocks(), block_size, block_size, ad_eq_class.get_matrix_CUDA_IA(), ad_eq_class.get_matrix_CUDA_JA(), ad_eq_class.get_matrix_CUDA_data(), NULL);
//AMGX_vector_upload(AMGX_vector_handle vec, int n, int block_dim,
//                  const void *data);
    AMGX_vector_upload(b_x, Nx*Ny, block_size, ad_eq_class.get_b_CUDA());
    AMGX_vector_upload(x_x, Nx*Ny, block_size, ad_eq_class.get_x_CUDA());
    check_memory("Arrays uploaded");   

    if(Nx*Ny<26)
    {
        AMGX_write_system(A_x, b_x, x_x, "some_system.mtx");
    }

    //int bsize_x, bsize_y, n, sol_size, sol_bsize;
    // AMGX_matrix_get_size(A_x, &n, &bsize_x, &bsize_y);
    // AMGX_vector_get_size(x_x, &sol_size, &sol_bsize);
    // printf("bsize_x %i, bsize_y %i, n %i, sol_size %i, sol_bsize %i \n", bsize_x, bsize_y, n, sol_size, sol_bsize);

    // //test!
    // cudaFree(SMatrix.data_d);
    // SMatrix.data_d = nullptr;

//solving....
    AMGX_solver_setup(solver_x, A_x);
    check_memory("AMGX solver setup"); 
    AMGX_solver_solve(solver_x, b_x, x_x);
    check_memory("AMGX solver solve"); 
    AMGX_solver_get_status(solver_x, &status_x);
//end solve

    AMGX_solver_destroy(solver_x);
    AMGX_vector_destroy(x_x);
    AMGX_vector_destroy(b_x);
    AMGX_matrix_destroy(A_x);
    check_memory("Arrays distroyed"); 
    AMGX_resources_destroy(resources_x);
    /* destroy config (need to use AMGX_SAFE_CALL after this point) */
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg_x));
    /* shutdown and exit */
    AMGX_SAFE_CALL(AMGX_finalize_plugins());
    AMGX_SAFE_CALL(AMGX_finalize());
    check_memory("AMGX distroyed"); 


//    check_memory("advect_diff_eq distroyed"); 
    

   
    return 0;
}