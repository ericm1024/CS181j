#pragma once

#include <array>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <type_traits>

#include <papi.h>

namespace papi_tools {

// A very thin wrapper around PAPI_strerror.
static inline void handle_papi_error(int papiReturnVal)
{
        if (papiReturnVal != PAPI_OK) {
                std::stringstream ss;
                ss << "PAPI error: " << PAPI_strerror(papiReturnVal)
                          << std::endl;
                throw std::runtime_error(ss.str());
        }
}

namespace detail {
class papi_lib_initializer {
        static bool lib_is_initialized_;
public:
        papi_lib_initializer()
        {
                if (!lib_is_initialized_) {
                        lib_is_initialized_ = true;
                        PAPI_library_init(PAPI_VER_CURRENT);
                }
        }
};

bool papi_lib_initializer::lib_is_initialized_ = false;


// is_unique: return true if the first template argument is the
// only argument of its value in the following parameter pack        
template <int...>
struct is_unique;
        
template <int event>
struct is_unique<event> : std::true_type {};

template <int event, int first_event, int... events>
struct is_unique<event, first_event, events...> {
        static constexpr bool value = event != first_event
                && is_unique<event, events...>::value;
};

// are_unique: return true if the parameter pack has no repeated values
template <int...>
struct are_unique;

template<>
struct are_unique<> : std::true_type {};

template <int event, int... events>
struct are_unique<event, events...> {
        static constexpr bool value = is_unique<event, events...>::value
                && are_unique<events...>::value;
};

// pack_contains: return true if a parameter pack contains an element
template <int...>
struct pack_contains;

template <int event>
struct pack_contains<event> : std::false_type {};

template <int event, int first_event, int... events>
struct pack_contains<event, first_event, events...> {
        static constexpr bool value = event == first_event
                || pack_contains<event, events...>::value;
};

};
        
template <int... events_>
class papi_event_set : detail::papi_lib_initializer {

        class papi_scoped_counter;
        friend class papi_scoped_counter;
        
        int set_{PAPI_NULL};
        std::array<long long, sizeof...(events_)> counts_ = {{0}};
        bool has_outstanding_counter_{false};

        // add_events: add each event in events_ to the event set. For implementaton
        // annoyances/details see this SO post:
        // http://stackoverflow.com/q/16758620/3775803
        template <int... events>
        typename std::enable_if<sizeof...(events) == 0>::type
        add_events()
        {}

        template <int event, int... events>
        void add_events()
        {
                int err = PAPI_add_event(set_, event);
                if (err != PAPI_OK) {
                        std::cerr << __func__ << ": failed to add event "
                                  << std::hex << event << std::endl;
                        handle_papi_error(err);
                }
                add_events<events...>();
        }

        // papi_scoped_counter: count papi events while this object is in
        // scope. see papi_event_set::counter()
        // 
        // this is an opaque type to the user
        class papi_scoped_counter {
                friend class papi_event_set;
                
                papi_event_set& es_;

        public:
                papi_scoped_counter(papi_event_set& es)
                        : es_{es}
                {
                        handle_papi_error(PAPI_accum(es_.set_,
                                                     es_.counts_.data()));
                        es_.has_outstanding_counter_ = true;
                }

                ~papi_scoped_counter()
                {
                        es_.has_outstanding_counter_ = false;
                        es_.counts_.fill(0);
                        handle_papi_error(PAPI_accum(es_.set_,
                                                     es_.counts_.data()));
                }
        };

        // count_impl: find the count for a given event by finding its index
        // in the parameter pack and retrieveing that value out of `counts_`
        template<int event>
        long long get_count_impl(size_t) const
        {
                throw std::runtime_error("attempt to count event not in set "
                                         "(this is a bug in papi_eventset)");
                return 0;
        }
        template <int event, int events_first, int... events_rest>
        long long get_count_impl(size_t idx) const
        {
                return event == events_first ? counts_.at(idx)
                        : get_count_impl<event, events_rest...>(idx + 1);
        }
        
public:
        papi_event_set()
                : set_{PAPI_NULL}
        {
                static_assert(detail::are_unique<events_...>::value,
                              "repeated events are not allowed");
                static_assert(sizeof...(events_) > 0,
                              "must have at least one event");
                
                handle_papi_error(PAPI_create_eventset(&set_));
                add_events<events_...>();
                handle_papi_error(PAPI_start(set_));
        }

        // no copy construction/assignment. basically
        // we just don't want more than one object owning an event set
        papi_event_set(const papi_event_set&) = delete;
        papi_event_set& operator=(const papi_event_set&) = delete;

        ~papi_event_set()
        {
                  handle_papi_error(PAPI_stop(set_, 0));
                  handle_papi_error(PAPI_cleanup_eventset(set_));
                  handle_papi_error(PAPI_destroy_eventset(&set_));
        }

        // get a scoped counter. Once the returned object goes out of scope,
        // call `count<event>` to get the number of occurances of that
        // event as measured in the counted scope.
        papi_scoped_counter scoped_counter()
        {
                if (has_outstanding_counter_) {
                        std::stringstream ss;
                        ss << __func__
                           << ": only one outstanding counter is allowed"
                           << std::endl;
                        throw std::runtime_error(ss.str());
                }
                return papi_scoped_counter(*this);
        }

        // query the number of occurances of an event as counted by the last
        // constructed counter
        template <int event>
        typename std::enable_if<detail::pack_contains<event, events_...>::value,
                                long long>::type
        get_count() const
        {
                return get_count_impl<event, events_...>(0);
        }
};

};
