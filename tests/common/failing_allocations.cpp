#include "failing_allocations.hpp"

#include <cstddef>
#include <cstdlib>
#include <new>

namespace {
thread_local long t_failIn = -1;
thread_local long t_made   = 0;
}  // namespace

namespace fv::test {

void failAllocationAfter(long count)
{
  t_failIn = count;
}

long allocationsMade()
{
  return t_made;
}

}  // namespace fv::test

void* operator new(std::size_t size)
{
  t_made++;
  if(t_failIn >= 0 && t_failIn-- == 0)
    throw std::bad_alloc();
  if(void* p = std::malloc(size ? size : 1))
    return p;
  throw std::bad_alloc();
}

void operator delete(void* p) noexcept
{
  std::free(p);
}

void operator delete(void* p, std::size_t) noexcept
{
  std::free(p);
}
