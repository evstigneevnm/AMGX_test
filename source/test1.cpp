#include <iostream>
#include <cuda_runtime.h>
#include <amgx_c.h>
#include <utils/cuda_safe_call.h>
#include <utils/cuda_support.h>
#include <utils/log.h>
#include <vector>
#include <map>

#include <matrix/block.h>
#include <matrix/row.h>
#include <matrix/sparse_matrix.h>

#include <amgx_wrap_resources.h>
#include <amgx_wrap_solver.h>
#include <advect_diff_eq2.h>

/*
run exmaple: in README.md
*/


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



int main(int argc, char const *argv[])
{
    
    typedef SCALAR_TYPE real;

    if(argc!=4)
    {
        printf("Usage: %s test_number problem_size path_to_amgx_config_file \n",argv[0]);
        return 0;
    }



    int Nx = 5, Ny = 5, Nz = 1;
    int test = atoi(argv[1]);
    if(test == 1)
    {
        Nx = atoi(argv[2]);
        Ny = Nx;
        Nz = 1;
        
    }
    std::string path_to_config_file( (char*)argv[3] );


    int device_number = init_cuda(4);
    if(device_number==-1)
    {
        std::cout << "error in GPU selection." << std::endl; 
        return 0;
    }

    real problem_size = 1*1*1;

    //typedefs
    const int block_size2 = 2;
    const int block_size3 = 3;
    
    typedef utils::log_std log_t;
    typedef block<block_size2, real> block2_t;
    typedef block<block_size3, real> block3_t;
    typedef row<block2_t> row2_t;
    typedef row<block3_t> row3_t;
    typedef numerical_algos::sparse_matrix<row2_t> sparse_matrix2_t;
    typedef numerical_algos::sparse_matrix<row3_t> sparse_matrix3_t;
    typedef advect_diff_eq2 <block2_t, row2_t, sparse_matrix2_t> adv_eq2_t;
    //typedef advect_diff_eq3 <block3_t, row3_t, sparse_matrix3_t> adv_eq3_t;
    typedef amgx_wrap::amgx_wrap_resources<real, log_t> amgx_wrap_resources_t;
    typedef amgx_wrap::amgx_wrap_solver<amgx_wrap_resources_t> amgx_wrap_t;

    // check_memory("init");
    log_t *log = new log_t();

    adv_eq2_t ad_eq_class(Nx, Ny);
    ad_eq_class.set_parameters(1.0e1, 100.0);

    ad_eq_class.form_CUDA_arrays();
    ad_eq_class.print_system();
    ad_eq_class.print_rows();

    printf("nnzb = %i\n",ad_eq_class.get_number_of_nonzero_blocks());
    
    // check_memory("adveciton_diffusion_class");
    amgx_wrap_resources_t* AMGX_R_p = new amgx_wrap_resources_t(log, 3, "AMGX:");
    amgx_wrap_t* AMGX = new amgx_wrap_t(AMGX_R_p, path_to_config_file, device_number);
    
    AMGX->set_problem_sizes(Nx*Ny, Nx*Ny, block_size2);
    AMGX->set_matrix_data(ad_eq_class.get_number_of_nonzero_blocks(), ad_eq_class.get_matrix_CUDA_IA(), ad_eq_class.get_matrix_CUDA_JA(), ad_eq_class.get_matrix_CUDA_data());
    AMGX->set_rhs_data(ad_eq_class.get_b_CUDA());
    AMGX->set_solution_data(ad_eq_class.get_x_CUDA());
    //AMGX->set_zero_solution(); //error?
    // check_memory("AMGX init done");
    //AMGX_matrix_upload_all(A_x, Nx*Ny, ad_eq_class.get_number_of_nonzero_blocks(), block_size2, block_size2, ad_eq_class.get_matrix_CUDA_IA(), ad_eq_class.get_matrix_CUDA_JA(), ad_eq_class.get_matrix_CUDA_data(), NULL);
    //AMGX_vector_upload(b_x, Nx*Ny, block_size2, ad_eq_class.get_b_CUDA());
    //AMGX_vector_upload(x_x, Nx*Ny, block_size2, ad_eq_class.get_x_CUDA());


    AMGX->upload_matrix();
    AMGX->upload_rhs();
    AMGX->upload_solution();


    int solve_res = AMGX->solve();
    switch(solve_res)
    {
        case(0):
            printf("converged\n");
            break;
        case(1):
            printf("failed\n");
            break;
        case(2):
            printf("not converged\n");
            break;
        case(3):
            printf("unknown error\n");
            break;
    }

    AMGX->download_solution();

    std::cout << "Residual norm = " << ad_eq_class.residual_norm_gpu() << std::endl;

    ad_eq_class.copy_results();
    ad_eq_class.print_system(true);

    AMGX->write_system("test_system.mtx");

    delete AMGX;

    // check_memory("AMGX distroyed"); 

    delete AMGX_R_p;
    delete log;
   
    return 0;
}
