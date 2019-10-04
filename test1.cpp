#include <iostream>
#include <cuda_runtime.h>
#include <amgx_c.h>
#include <utils/cuda_safe_call.h>
#include <utils/cuda_support.h>
#include <vector>
#include <map>

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

template<class Block>
class advect_diff_eq
{
public:
    typedef typename Block::Block_T T;
    static const unsigned int Block_Size = Block::Block_Size;

    advect_diff_eq(int Nx_, int Ny_):
    Nx(Nx_), Ny(Ny_)
    {
        b = (double*)malloc(Nx*Ny*Block_Size*sizeof(T));
        x = (double*)malloc(Nx*Ny*Block_Size*sizeof(T));

    }
    ~advect_diff_eq()
    {

        free_C_array(b);
        free_C_array(x);

    }

    T* b = nullptr;
    T* x = nullptr;

private:
    int Nx, Ny;
    void initial_vectors()
    {
        

    }
    void free_C_array(T*& array)
    {
        if(array!=nullptr)
        {
            free(array);    
            array = nullptr;
        }
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
    

    if(argc!=2)
    {
        printf("Usage: %s path_to_amgx_config_file \n",argv[0]);
        return 0;
    }

    AMGX_Mode mode_x;
    AMGX_config_handle cfg_x;
    AMGX_resources_handle resources_x;
    AMGX_matrix_handle A_x;
    AMGX_vector_handle b_x, x_x;
    AMGX_solver_handle solver_x;
    AMGX_SOLVE_STATUS status_x;



    if(init_cuda(10)==-1)
    {
        std::cout << "error in GPU selection." << std::endl; 
        return 0;
    }

  
    const int block_size = 2;
    typedef block<block_size, double> block_t;
    typedef row<block_t> row_t;

    block_t Bd, B;
    row_t R0;
    int Nx = 3000, Ny = 3000;


    sparse_matrix<row_t> SMatrix(Nx*Ny);
    double *b_h = (double*)malloc(Nx*Ny*block_size*sizeof(double));
    double *x_h = (double*)malloc(Nx*Ny*block_size*sizeof(double));
    double *b_d, *x_d;
    b_d =  device_allocate<double>(Nx*Ny*block_size);
    x_d =  device_allocate<double>(Nx*Ny*block_size);

    for(int j=0;j<Nx;j++)
    { 
        for(int k=0;k<Ny;k++)
        {
            int ind = (j)*Ny+k;
            int ind_km = (j)*Ny + (k-1);
            int ind_kp = (j)*Ny + (k+1);
            int ind_jm = (j-1)*Ny + (k);
            int ind_jp = (j+1)*Ny + (k);

            b_h[2*ind] = double(1.0);
            b_h[2*ind+1] = double(2.0);

            x_h[2*ind] = double(0.0);
            x_h[2*ind+1] = double(0.0);

            Bd.set_block({double(8),-1,-0.2,double(8)});
            
            R0.set_reserve_row(ind, 5);
            if(k>0)
            {
                B.set_block({2,0,0,2});
                R0.add_block(B, ind_km);
            }
            if(j>0)
            {
                B.set_block({1,0,0,1});
                R0.add_block(B, ind_jm);
            }
            if(k<Ny-1)
            {
                B.set_block({2,0.3,0,2});
                R0.add_block(B, ind_kp);
            }
            if(j<Nx-1)
            {
                B.set_block({1,0,1,1});
                R0.add_block(B, ind_jp);
            }
            // if(k<Ny-1)
            // {
            //     Bp.set_block({1,0,0,1});
            //     R0.add_block(Bp, ind_p);      
            // }
            R0.add_block(Bd, ind); //diagonal block
            
            SMatrix.add_row(R0);

        }
    }
    
    host_2_device_cpy<double>(b_d, b_h, Nx*Ny*block_size);
    host_2_device_cpy<double>(x_d, x_h, Nx*Ny*block_size);


    check_memory("init");
    SMatrix.form_matrix_gpu();
    if(Nx*Ny<26)
    {
        SMatrix.print_matrix();
    }

    printf("nnzb = %i\n",SMatrix.get_number_of_nonzero_blocks());

    check_memory("Smatrix");
  
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    AMGX_SAFE_CALL(AMGX_install_signal_handler());
    set_amgx_mode<double>(mode_x);

    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg_x, argv[1]));
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
    
    AMGX_matrix_upload_all(A_x, Nx*Ny, SMatrix.get_number_of_nonzero_blocks(), block_size, block_size, SMatrix.IA_d, SMatrix.JA_d, SMatrix.data_d, NULL);
//AMGX_vector_upload(AMGX_vector_handle vec, int n, int block_dim,
//                  const void *data);
    AMGX_vector_upload(b_x, Nx*Ny, block_size, b_d);
    AMGX_vector_upload(x_x, Nx*Ny, block_size, x_d);
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

    cudaFree(b_d);
    cudaFree(x_d);

    check_memory("Arrays distroyed"); 
    free(b_h);
    free(x_h);



   
    return 0;
}