// Dispatcher for `merge_and_collect_semisort`. Two variants live in
// dedicated headers; this is the single translation unit that picks
// one and pulls its definition into the static lib.
//
// Default = secondary-hash equality (h1 + h2 match).
// PE_SEMISORT_SOUND → structural equality via sigs_equal.

#ifdef PE_SEMISORT_SOUND
#include "parallel_egraph/semisort_sound.hpp"
#else
#include "parallel_egraph/semisort_secondary.hpp"
#endif
