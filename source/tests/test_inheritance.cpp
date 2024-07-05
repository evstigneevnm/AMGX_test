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


#include <iostream>
#include <vector>

template<class T>
class Base
{
public:
    int m_public; // can be accessed by anybody
    T* x_h = nullptr;
    size_t size;
    
    Base(size_t size_):
    size(size_)
    {
        x_h=(T*)malloc(sizeof(T)*size);
    }
    ~Base()
    {
        if(x_h!=nullptr)
            free(x_h);
    }
    void test()
    {
        std::cout << "test";
    }
    
};
 
template<class T>
class Derived: public Base<T>
{
public:
    using Base<T>::m_public;
    using Base<T>::x_h;
    using Base<T>::test;
    
    size_t size;
    Derived(size_t size_):
    size(size_),
    Base<T>(size)
    {
        m_public = 1; // allowed: can access public base members from derived class
        
    }
    void print()
    {
        for(int j=0;j<size;j++)
            x_h[j]=0.0;
        
        test();
        std::cout << m_public << std::endl;
    }
};

int main()
{
    
    Derived<double> d(100);
    d.print();
    return 0;
}
