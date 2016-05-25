// -*- C++ -*-

// Many of the homework assignments have definitions and includes that
//  are common across several executables, so we group them together.
#include "CommonDefinitions.h"
#include <functional>
#include <cmath>
#include <iostream>

#include "../papi_tools.hpp"

using namespace papi_tools;

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::vector;
using std::string;
using std::array;
using std::cout;
using std::endl;

int main()
{
        papi_event_set<PAPI_FP_INS> event_set;
        size_t size = 1 << 25;
        vector<double> v(size, 0);
        
        std::generate(v.begin(), v.end(),
                      std::bind(std::uniform_real_distribution<double>(0, 1),
                                std::mt19937()));

        const auto tic = high_resolution_clock::now();
        {
                const auto c_ = event_set.scoped_counter();
                const size_t end = v.size() - 10;
                for (size_t i = 0; i < end; ++i) {
                        v[i] = std::sin(v[i]) *
                                std::sin(v[i+1]) *
                                std::sin(v[i+2]) *
                                std::sin(v[i+3]) *
                                std::sin(v[i+4]) *
                                std::sin(v[i+5]) *
                                std::sin(v[i+6]) *
                                std::sin(v[i+7]) *
                                std::sin(v[i+8]) *
                                std::sin(v[i+9]) *
                                std::sin(v[i+10]);
                }
        }
        const auto toc = high_resolution_clock::now();
        const double elapsed = duration_cast<duration<double>>(toc - tic).count();

        cout << "elapsed time: " << elapsed << ", flops: "
             << event_set.get_count<PAPI_FP_INS>() << endl;
        cout << "flops/second: "
             << event_set.get_count<PAPI_FP_INS>()/elapsed << endl;
}
