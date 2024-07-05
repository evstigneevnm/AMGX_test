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

#pragma once

#include <iostream>
#include <vector>


template<class Block>
class row
{
public:
    typedef typename Block::Block_T T;
    static const unsigned int Block_Size = Block::Block_Size;


    row(){ row_number = 0; number_of_nonzero_blocks = 0; }
    ~row(){}
    
    void set_reserve_row(int row_l, int nnz_reserve_l)
    {
        set_row(row_l);
        reserve_nnz(nnz_reserve_l);
    }

    void set_row(int row_l)
    {
        reset_data();
        row_number = row_l;

    }
    
    void add_block(Block block_l, int col_l)
    {
        if( block_l.is_set() )
        {
            data.push_back({col_l, block_l});
            //column.push_back(col_l);
            number_of_nonzero_blocks++;
        }
        else
        {

        }
    }
    void print_row()
    {
        std::cout << "row number = " << row_number << std::endl;
        for(auto &x: data)
        {
            std::cout << "block_column: " << x.first << std::endl;
            x.second.print_block();
        }
        std::cout << "nnz = " << number_of_nonzero_blocks << std::endl;
    }
    void reserve_nnz(int nnz_reserve_l)
    {   
        data.reserve(nnz_reserve_l);
        //column.reserve(nnz_reserve_l);
    }

    int get_number_of_nonzero_blocks()
    {
        return number_of_nonzero_blocks;
    }
    int get_row_number()
    {
        return row_number;
    }

    std::vector<std::pair<int, Block>> data;

private:
    int number_of_nonzero_blocks;
    int row_number;

    void reset_data()
    {
        number_of_nonzero_blocks = 0;
        data.clear();
        //column.clear();
    }


};
