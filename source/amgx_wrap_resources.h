/*
 AMGX_test is a program for block advection-diffusion tests of AMGX solver from NVIDIA
Copyright (C) Dr.Evstigneev N.M.
This file is part of AMGX_test.
AMGX_test is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

AMGX_test is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#ifndef __AMGX_WRAP_RESOURCES_H__
#define __AMGX_WRAP_RESOURCES_H__


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


template<typename T, class Log>
class amgx_wrap_resources: public utils::logged_obj_base<Log>
{
public:
    typedef utils::logged_obj_base<Log> logged_obj_t;
    typedef T scalar_T;

    amgx_wrap_resources(Log *log_p_, int obj_log_lev, const std::string &log_msg_prefix):
    utils::logged_obj_base<Log>(log_p_, obj_log_lev, log_msg_prefix)
    {
        
        AMGX_SAFE_CALL(AMGX_initialize());
        AMGX_SAFE_CALL(AMGX_initialize_plugins());
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
        //AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
        set_amgx_mode<T>(mode_x); 
        logged_obj_t::info_f("resorces initialized");

    }
#if USE_MPI == 1
    amgx_wrap_resources(Log *log_p_, int obj_log_lev, const std::string &log_msg_prefix, MPI_Comm*& amgx_mpi_comm_p_):
    utils::logged_obj_base<Log>(log_p_, obj_log_lev, log_msg_prefix),
    amgx_mpi_comm_p(amgx_mpi_comm_p_)
    {
        
        AMGX_SAFE_CALL(AMGX_initialize());
        AMGX_SAFE_CALL(AMGX_initialize_plugins());
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
        AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
        set_amgx_mode<T>(mode_x); 
        logged_obj_t::info_f("resorces initialized");

    }
#endif

    ~amgx_wrap_resources()
    {
        AMGX_SAFE_CALL(AMGX_finalize_plugins());
        AMGX_SAFE_CALL(AMGX_finalize());
        logged_obj_t::info_f("resorces removed");
    }
    

    AMGX_Mode get_mode()
    {
        return mode_x;
    }

#if USE_MPI == 1    
    MPI_Comm* get_MPI_Comm()
    {
        return amgx_mpi_comm_p;
    }
#endif

private:

    AMGX_Mode mode_x;
#if USE_MPI == 1    
    MPI_Comm* amgx_mpi_comm_p;
#endif
    

    void print_callback(const char *msg, int length)
    {

        logged_obj_t::info_f(msg);
    }



};


}




#endif
