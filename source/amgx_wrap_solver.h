#ifndef __AMGX_WRAP_SOLVER_H__
#define __AMGX_WRAP_SOLVER_H__

#include <string>
//this config sets usage of MPI
#include <HPC_global_config.h>
//assume that -I includes AMGX include directory!
#include <amgx_c.h>
#include <utils/logged_obj_base.h>

#if USE_MPI == 1
    #include <mpi.h>
#endif

namespace amgx_wrap
{

template<class AMGXResources>
class amgx_wrap_solver
{
public:
    typedef typename AMGXResources::scalar_T T;

    amgx_wrap_solver(AMGXResources *amgx_r_p_, std::string path_to_config_file, int device_number_, bool create_solver_ = true):
    amgx_r_p(amgx_r_p_),
    device_number(device_number_)
    {
        constructor_common(path_to_config_file, create_solver_);
    }

    amgx_wrap_solver(AMGXResources *amgx_r_p_, std::string path_to_config_file, int device_number_, size_t rows_, size_t cols_, int block_size_,  bool create_solver_ = true):
    amgx_r_p(amgx_r_p_),
    device_number(device_number_)
    {
        constructor_common(path_to_config_file, create_solver_);
        set_problem_sizes(rows_, cols_, block_size_);
    }


    ~amgx_wrap_solver()
    {
        if(AMXG_axpy_created)
        {      
            AMGX_vector_destroy(x_x);
            AMGX_vector_destroy(b_x);
            AMGX_matrix_destroy(A_x);
        }
        if(solver_created)
        {
            AMGX_solver_destroy(solver_x);
        }

        AMGX_resources_destroy(resources_x);
        AMGX_SAFE_CALL(AMGX_config_destroy(cfg_x));
    }
    

    void create_solver()
    {
        if(!solver_created)
        {
            AMGX_solver_create(&solver_x, resources_x, amgx_r_p->get_mode(), cfg_x);
            solver_created = true;
        }
    }


    void create_solver_substitute_config(std::string path_to_config_file)
    {
        if(!solver_created)
        {
            AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg_x1, path_to_config_file.c_str() ));
            AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg_x1, "exception_handling=1"));
            AMGX_solver_create(&solver_x, resources_x, amgx_r_p->get_mode(), cfg_x1);
            solver_created = true;
 
        }
    }

    void set_problem_sizes(size_t rows_, size_t cols_, int block_size_)
    {
        rows = rows_;
        cols = cols_;
        block_size = block_size_;

    }

    void set_matrix_data(size_t nnz_blocks_, const int* row_p_, const int* col_p_, const T* data_, const T* data_diag_ = NULL)
    {
        nnz_blocks = nnz_blocks_;
        row_p = row_p_;
        col_p = col_p_;
        data_A = data_;
        data_A_diag = data_diag_;

    }
    void set_rhs_data(const T* data_)
    {
        data_b = data_;
    }
    void set_solution_data(T* data_)
    {
        data_x = data_;
    }

    
//*** upload matrix
    void upload_matrix(size_t rows_, size_t cols_, size_t nnz_blocks_, int block_size_, const int*& row_p_, const int*& col_p_, const T*& data_, const T*& data_diag_ = NULL)
    {
        nnz_blocks = nnz_blocks_;   
        AMGX_matrix_upload_all(A_x, rows_, nnz_blocks, block_size_, block_size_, row_p_, col_p_, data_, data_diag_);

    }



    void upload_matrix(size_t nnz_blocks_, const int*& row_p_, const int*& col_p_, const T*& data_, const T*& data_diag_ = NULL)
    {
        upload_matrix(rows, cols, nnz_blocks_, block_size, row_p_, col_p_, data_, data_diag_);
    }

    void upload_matrix()
    {
        upload_matrix(nnz_blocks, row_p, col_p, data_A, data_A_diag);
    }


//*** update matrix
    void update_matrix(const T*& data_, const T*& data_diag_ = NULL)
    {
        AMGX_matrix_replace_coefficients(A_x, rows, nnz_blocks, data_, data_diag_);
    }

    void write_system(const std::string& file_name)
    {
        AMGX_write_system(A_x, b_x, x_x, file_name.c_str() );
    }

    void update_matrix()
    {
        update_matrix(data_A, data_A_diag);
    }

//*** upload rhs
    void upload_rhs(size_t cols_, int block_size_, const T*& data_)
    {
        AMGX_vector_upload(b_x, cols_, block_size_, data_);
    }

    void upload_rhs(const T*& data_)
    {
        upload_rhs(cols, block_size, data_);
    }

    void upload_rhs()
    {
        upload_rhs(data_b);
    }

//*** upload-zero-download solution
    void upload_solution(size_t cols_, int block_size_, T*& data_)
    {
        AMGX_vector_upload(x_x, cols_, block_size_, data_);
    }

    void upload_solution(T*& data_)
    {
        upload_solution(cols, block_size, data_);
    }

    void upload_solution()
    {
        upload_solution(data_x);
    }

    void set_zero_solution(size_t cols_, int block_size_)
    {
        AMGX_vector_set_zero(x_x, cols_, block_size_);
    }
    void set_zero_solution()
    {
        set_zero_solution(cols, block_size);
    }

    void download_solution(T* data_)
    {
        AMGX_vector_download(x_x, data_);
    }

    void download_solution()
    {
        download_solution(data_x);
    }



//*** execute solver: analize, execute and check status
//    0 - normal termination
//    1 - failed
//    2 - not converged
//    3 - unknown error
    int solve()
    {
        int return_code = 0;
        AMGX_solver_setup(solver_x, A_x);
        AMGX_solver_solve(solver_x, b_x, x_x);
        AMGX_solver_get_status(solver_x, &status_x);
        switch(status_x)
        {
            case AMGX_SOLVE_SUCCESS:
                return_code = 0;
                break;
            case AMGX_SOLVE_FAILED:
                return_code = 1;
                break;
            case AMGX_SOLVE_DIVERGED:
                return_code = 2;
                break;
            default:
                return_code = 3;
                break;
        }

        return(return_code);
    }

//*** writes linear system to file in matrix market format.
//    Should be used only for debug!
    void save_system_to_file(std::string matrix_file)
    {
        AMGX_write_system(A_x, b_x, x_x, matrix_file.c_str());
    }


private:
    bool AMXG_axpy_created = false;
    bool solver_created = false;

    AMGXResources* amgx_r_p;

    const int* row_p = NULL;
    const int* col_p = NULL;
    const T* data_A = NULL;
    const T* data_A_diag = NULL;
    T* data_x = NULL;
    const T* data_b = NULL;

    size_t rows;
    size_t cols;
    size_t block_size;
    size_t nnz_blocks;

    AMGX_config_handle cfg_x;
    AMGX_config_handle cfg_x1;
    AMGX_resources_handle resources_x;
    AMGX_matrix_handle A_x;
    AMGX_vector_handle b_x, x_x;
    AMGX_solver_handle solver_x;
    AMGX_SOLVE_STATUS status_x;

    int device_number;


    void constructor_common(std::string path_to_config_file, bool create_solver_)
    {
        AMXG_axpy_created = false;
        solver_created = false;
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg_x, path_to_config_file.c_str() ));
        AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg_x, "exception_handling=1"));

        #if USE_MPI == 1  
            AMGX_resources_create(&resources_x, cfg_x, amgx_r_p->get_MPI_Comm(), 1, &device_number);

        #else
            AMGX_resources_create(&resources_x, cfg_x, NULL, 1, &device_number);
        #endif
    

        AMGX_matrix_create(&A_x, resources_x, amgx_r_p->get_mode());
        AMGX_vector_create(&x_x, resources_x, amgx_r_p->get_mode());
        AMGX_vector_create(&b_x, resources_x, amgx_r_p->get_mode());
        AMXG_axpy_created = true;
        
        if(create_solver_)
        {
            AMGX_solver_create(&solver_x, resources_x, amgx_r_p->get_mode(), cfg_x);
            solver_created = true;
        }

    }


};

}


#endif