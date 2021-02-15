#pragma once

#include <iostream>

template<unsigned int Block_Size_I, class T>
class block
{
public:    
    static const unsigned int Block_Size = Block_Size_I;
    typedef T Block_T;

    block(){ index = 0; }
    ~block(){}

    //function sets a block in row major format
    //if initializer_list is not equal to block Block_Size then error is produced TODO!
    void set_block( std::initializer_list<T> list )
    {
        
        if( list.size() == Block_Size*Block_Size )
        {
            reset_index();
            for( auto elem : list )
            {
                data[index++] = T(elem);
            }
        }
        else
        {
            //TODO produce error!
            std::cerr << "set_block(list): block Block_Size is not consistent with list Block_Size. Block is not set." << std::endl;
        }    
    }
    //funciton adds to block by any number of parameters withought reseting the index
    void add_to_block( std::initializer_list<T> list )
    {
        for( auto elem : list )
        {
            if(index<Block_Size*Block_Size)
            {
                data[index++] = T(elem);
            }
            else
            {
                //TODO produce error!
                std::cerr << "add_to_block(list): list not added since commulative list Block_Size is greater than added data to block." << std::endl;
            }
        }
    }

    //function updates block data by setting the whole addition block
    void update_set_block(std::initializer_list<T> list)
    {
        if( list.size() == Block_Size*Block_Size )
        {
            reset_index();
            for( auto elem : list )
            {
                data[index++] += T(elem);
            }
        }
        else
        {
            //TODO produce error!
            std::cerr << "update_block(list): block Block_Size is not consistent with list Block_Size. Block is not updated." << std::endl;
        }    

    }
    //funciton updates to block by any number of parameters assuming that index is already reached block_size*block_size-1. The index is reseted if the block was set before. Otherwise this function does nothing and reports a error
    void update_add_to_block( std::initializer_list<T> list )
    {
        if(index == Block_Size*Block_Size-1)
        {
            index = 0;
        }
        for( auto elem : list )
        {
            if(index<Block_Size*Block_Size)
            {
                data[index++] = T(elem);
            }
            else
            {
                //TODO produce error!
                std::cerr << "update_add_to_block(list): list not added since commulative list Block_Size is greater than added data to block." << std::endl;
            }
        }
    }
    void reset_block()
    {
        reset_index();
        for(int j=0;j<Block_Size*Block_Size;j++)
            data[j]=T(0);
    }

    T data[Block_Size*Block_Size];
    int index = 0;
    unsigned int size = Block_Size;

    void print_block()
    {
        for(int j=0;j<Block_Size;j++)
        {
            for(int k=0;k<Block_Size;k++)
            {
                std::cout << data[j*Block_Size+k] << " ";
            }
            std::cout << "|" << std::endl;
        }
    }
    bool is_set()
    {
        if(index == (Block_Size*Block_Size) )
            return true;
        else
            return false;
    }

private:
    void reset_index()
    {
        index = 0;


    }

};

