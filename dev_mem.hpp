#pragma once
#include <cstddef>
#include <stdexcept>

template <typename T>
class dev_mem {
public:
        using value_type = T;
        using pointer = T*;

private:
        pointer mem_;
        std::size_t n_; // number of entries in mem_, *NOT* its size in bytes

        void do_free() const
        {
                checkCudaError(cudaFree((void*)mem_));
        }

        void do_alloc()
        {
                checkCudaError(cudaMalloc((void **)&mem_, n_*sizeof(T)));
        }
        
public:
        explicit dev_mem(std::size_t n)
                : n_{n}
        {
                do_alloc();
        }

        // DATA IS HOST MEMORY
        dev_mem(pointer data, std::size_t n)
                : dev_mem{n}
        {
                checkCudaError(cudaMemcpy((void*)mem_, data, n * sizeof(T),
                                          cudaMemcpyHostToDevice));
        }

        ~dev_mem()
        {
                do_free();
        }

        dev_mem(const dev_mem& other)
                : dev_mem{other.n_}
        {
                checkCudaError(cudaMemcpy((void*)mem_, other.mem_,
                                          n_ * sizeof(T),
                                          cudaMemcpyDeviceToDevice));
        }

        dev_mem& operator=(const dev_mem& other)
        {
                do_free();
                n_ = other.n_;
                do_alloc();
                checkCudaError(cudaMemcpy((void*)mem_, other.mem_,
                                          n_ * sizeof (T),
                                          cudaMemcpyDeviceToDevice));
                return *this;
        }

        dev_mem(const dev_mem&& other)
                : n_{other.n_}, mem_{other.mem_} {}
        
        dev_mem& operator=(const dev_mem&& other)
        {
                do_free();
                n_ = other.n_;
                mem_ = other.mem_;
                return *this;
        }

        pointer get() const {return mem_;}
        operator pointer() const {return mem_;} // yummy
        std::size_t size() const {return n_;}

        // OUT IS HOST MEMORY
        void write_to(pointer out, std::size_t size) const
        {
                if (size != n_)
                        throw std::length_error("dev_mem::write_to: bad size");

                checkCudaError(cudaMemcpy(out, mem_, size*sizeof(T),
                                          cudaMemcpyDeviceToHost));
        }
};

// use:
// 
//     auto dm = make_dev_mem(ptr, size);
// 
// instead of
//
//     dev_mem<double> dm{ptr, size};
//
// to avoid specifying template type when it can be infered (because C++
// doesn't allow inferance of class template arguments from ctor args
template<typename T>
dev_mem<T> make_dev_mem(T* data, std::size_t n)
{
        return dev_mem<T>{data, n};
}
