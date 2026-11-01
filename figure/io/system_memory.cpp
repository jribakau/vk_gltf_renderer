#include <figure/io/system_memory.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace fv::io {

SystemMemory systemMemory()
{
  SystemMemory out;
#ifdef _WIN32
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if(GlobalMemoryStatusEx(&status))
  {
    out.totalPhysical     = status.ullTotalPhys;
    out.availablePhysical = status.ullAvailPhys;
    out.availableCommit   = status.ullAvailPageFile;
    out.known             = true;
  }
#endif
  return out;
}

}  // namespace fv::io
