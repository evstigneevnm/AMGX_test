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
            reset_data();
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
            std::cout << std::endl;
        }
    }

private:
    void reset_data()
    {
        index = 0;


    }

};

