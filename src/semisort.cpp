// Dispatcher for the two `merge_and_collect_semisort` impls. Each impl
// lives in its own header (semisort_ordered.hpp / semisort_hash.hpp);
// this is the single translation unit that picks one and pulls its
// definition into the static lib. The CMake-side conditional source
// selection that this used to require is now handled by the #ifdef
// below, keeping CMakeLists uniform.
//
// Default = ordered (integer_sort + dual-hash CanonEntry).
// PE_GROUPBY_HASH=ON  → hash kernel (group_by_key + sigs_equal).

#ifdef PE_GROUPBY_HASH
#include "parallel_egraph/semisort_hash.hpp"
#else
#include "parallel_egraph/semisort_ordered.hpp"
#endif
