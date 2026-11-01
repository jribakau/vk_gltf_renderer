#pragma once

// One allocation made to fail, on the thread that asks: what runs out of memory can then be tested at
// each allocation it makes. The test program's operator new does it.

namespace fv::test {

// The `count`th allocation from now on this thread throws std::bad_alloc, once; below zero: none.
void failAllocationAfter(long count);
// How many this thread has made.
long allocationsMade();

}  // namespace fv::test
