#include <algorithm>
#include <vector>
#include <random>
#include <functional>
#include <limits>
#include <fstream>
#include <chrono>
#include <numeric>
#include <string>
#include <map>
#include <set>

#include "./cpp-btree-master/btree_set.h"
#include "../papi_tools.hpp"
#include "../Utilities.h"

using namespace papi_tools;

using std::vector;
using std::string;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

int main()
{
        using test_t = long;
        const auto test_t_generator = std::bind(std::uniform_int_distribution<test_t>(
                                                        std::numeric_limits<test_t>::min(),
                                                        std::numeric_limits<test_t>::max()),
                                                std::mt19937());

        papi_event_set<PAPI_L1_DCM> event_set;
        const auto size = 1_z << 20;
        const auto runs = 5;
        std::fstream data_file("data/Main1_data.csv", std::ios_base::out | std::ios_base::trunc);

        data_file << "node_size, "
                  << "insert_cache_misses, insert_time, "
                  << "iterate_cache_misses, iterate_time, "
                  << "erase_cache_misses, erase_time"
                  << std::endl;

        const auto bench_one_size = [&](auto set) {
                std::multimap<std::string, std::pair<long long, double>> data;

                std::cout << "run: ";
                
                for (int i = 0; i < runs; ++i) {
                        set.clear();
                        vector<test_t> v(size, 0);
                        std::generate(v.begin(), v.end(), test_t_generator);

                        const auto do_one_method = [&](const string& name,
                                                       const auto& method) {

                                Utilities::clearCpuCache();

                                const auto tic = high_resolution_clock::now();
                                {
                                        const auto c_ = event_set.scoped_counter();
                                        method();
                                }
                                const auto toc = high_resolution_clock::now();

                                data.insert(std::make_pair(name,
                                                           std::make_pair(event_set.get_count<PAPI_L1_DCM>(),
                                                                          duration_cast<duration<double>>(toc - tic).count())));
                        };

                        do_one_method("insert", [&]() {
                                        for (const auto e : v)
                                                set.insert(e);
                                });

                        do_one_method("iterate", [&]() {
                                        for (const auto e : set)
                                                (void)e;
                                });

                        do_one_method("erase", [&]() {
                                        for (const auto e : v)
                                                set.erase(e);
                                });

                        std::cout << i << std::flush;
                }

                std::cout << std::endl;

                for (const auto& test_name : {"insert", "iterate", "erase"}) {
                        auto results = data.equal_range(test_name);
                        std::pair<long long, double> sums =
                                std::accumulate(results.first, results.second, std::make_pair(0LL, double(0)),
                                                [](const auto& lhs, const auto& rhs) {
                                                        // I literally hate this language sometimes
                                                        return std::make_pair(lhs.first + rhs.second.first,
                                                                              lhs.second + rhs.second.second);
                                                });

                        long long avg_cache_misses = sums.first / runs;
                        double avg_time = sums.second / runs;

                        data_file << ", " << avg_cache_misses << ", " << avg_time;
                }

                data_file << std::endl;
        };

        // UGH, why can't you iterate through a list of constexprs
#define one_size(_sz) do {                                              \
                constexpr size_t sz = (_sz);                            \
                data_file << sz;                                        \
                std::cout << "About to do size " << sz << std::endl;    \
                bench_one_size(btree::btree_set<test_t, std::less<test_t>, std::allocator<test_t>, sz>()); \
        } while(0)

        // run the test with a bunch of different node sizes
        data_file << 1;
        std::cout << "About to do size 1" << std::endl;
        bench_one_size(std::set<test_t>());
        one_size(1 << 4);
        one_size(1 << 5);
        one_size(1 << 6);
        one_size(1 << 7);
        one_size(1 << 8);
        one_size(1 << 9);
        one_size(1 << 10);
        one_size(1 << 11);
        one_size(1 << 12); // 4096
}
