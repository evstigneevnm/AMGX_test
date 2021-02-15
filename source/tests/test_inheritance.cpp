/******************************************************************************

                              Online C++ Compiler.
               Code, Compile, Run and Debug C++ program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

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
