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

#include <advect_diff_eq.h>

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
    ad_eq_class.print_system();

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