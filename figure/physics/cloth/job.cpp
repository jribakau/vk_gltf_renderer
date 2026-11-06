#include <figure/physics/cloth/job.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <type_traits>

namespace fv::physics::cloth {

namespace {

// "FVCJOB" and a version: a file of another version isn't read. Version 2 adds the drawn skin
// (Surface::skinIndex8 and the dual quaternion share); a version 1 file is read without it, its
// surfaces skinned with the four as when it was written. Version 3 adds a surface's capsule floor
// (Surface::capsules); an older file is read without one.
constexpr char     kMagic[8]   = {'F', 'V', 'C', 'J', 'O', 'B', '0', '3'};
constexpr char     kMagicV2[8] = {'F', 'V', 'C', 'J', 'O', 'B', '0', '2'};
constexpr char     kMagicV1[8] = {'F', 'V', 'C', 'J', 'O', 'B', '0', '1'};
constexpr uint64_t kMostItems = uint64_t(1) << 32;

struct Writer
{
  std::ofstream& o;
  template <class T>
  void value(const T& v)
  {
    static_assert(std::is_trivially_copyable_v<T>);
    o.write(reinterpret_cast<const char*>(&v), sizeof v);
  }
  template <class T>
  void list(const std::vector<T>& v)
  {
    value(uint64_t(v.size()));
    if(!v.empty())
      o.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
  }
  void text(const std::string& s)
  {
    value(uint64_t(s.size()));
    o.write(s.data(), std::streamsize(s.size()));
  }
};

struct Reader
{
  std::ifstream& in;
  bool           ok = true;
  template <class T>
  void value(T& v)
  {
    static_assert(std::is_trivially_copyable_v<T>);
    ok = ok && bool(in.read(reinterpret_cast<char*>(&v), sizeof v));
  }
  template <class T>
  void list(std::vector<T>& v)
  {
    uint64_t n = 0;
    value(n);
    ok = ok && n < kMostItems;
    if(!ok)
      return;
    v.resize(size_t(n));
    if(n)
      ok = bool(in.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T))));
  }
  void text(std::string& s)
  {
    uint64_t n = 0;
    value(n);
    ok = ok && n < kMostItems;
    if(!ok)
      return;
    s.resize(size_t(n));
    ok = bool(in.read(s.data(), std::streamsize(n)));
  }
};

// Every field of a surface, in one order for both ways.
template <class IO, class S>
void surface(IO& io, S& m, int version)
{
  io.list(m.rest);
  io.list(m.posed);
  io.list(m.tris);
  io.list(m.polyStart);
  io.list(m.polyVerts);
  io.value(m.kind);
  for(auto* v : {&m.strength, &m.stretch, &m.shear, &m.bend, &m.buckling, &m.bucklingRatio, &m.density, &m.contraction, &m.damping, &m.friction, &m.offset})
    io.list(*v);
  io.list(m.layer);
  io.list(m.selfCollide);
  io.list(m.collide);
  io.list(m.visible);
  io.value(m.gravity);
  io.list(m.skinIndex);
  io.list(m.skinWeight);
  io.list(m.bones);
  io.list(m.restLocal);
  io.value(m.toSpace);
  if(version >= 2)
  {
    io.list(m.skinIndex8);
    io.list(m.skinWeight8);
    io.value(m.dualQuat);
    io.list(m.dualQuatShare);
  }
  if(version >= 3)
    io.list(m.capsules);
}

template <class IO, class O>
void options(IO& io, O& o)
{
  io.value(o.air);
  io.value(o.frameDt);
  io.value(o.substeps);
  io.value(o.iterations);
  io.value(o.stretchSweeps);
  io.value(o.reachLinks);
  io.value(o.transition);
  io.value(o.maxHold);
  io.value(o.recoverLost);
  io.value(o.restMove);
}

}  // namespace

bool writeJob(const std::filesystem::path& file, const Job& job)
{
  std::ofstream o(file, std::ios::binary);
  if(!o)
    return false;
  Writer w{o};
  o.write(kMagic, sizeof kMagic);
  w.value(uint64_t(job.surfaces.size()));
  for(const Surface& m : job.surfaces)
    surface(w, m, 3);
  options(w, job.options);
  w.value(uint64_t(job.drapes.size()));
  for(const auto& [slot, name] : job.drapes)
  {
    w.value(int32_t(slot));
    w.text(name);
  }
  return bool(o);
}

std::optional<Job> readJob(const std::filesystem::path& file)
{
  std::ifstream in(file, std::ios::binary);
  char          magic[sizeof kMagic]{};
  if(!in || !in.read(magic, sizeof magic))
    return std::nullopt;
  const int version = std::memcmp(magic, kMagic, sizeof magic) == 0   ? 3
                      : std::memcmp(magic, kMagicV2, sizeof magic) == 0 ? 2
                      : std::memcmp(magic, kMagicV1, sizeof magic) == 0 ? 1
                                                                        : 0;
  if(version == 0)
    return std::nullopt;
  Reader   r{in};
  Job      job;
  uint64_t n = 0;
  r.value(n);
  if(!r.ok || n > 4096)
    return std::nullopt;
  job.surfaces.resize(size_t(n));
  for(Surface& m : job.surfaces)
    surface(r, m, version);
  options(r, job.options);
  r.value(n);
  if(!r.ok || n > 4096)
    return std::nullopt;
  for(uint64_t k = 0; k < n && r.ok; k++)
  {
    int32_t     slot = 0;
    std::string name;
    r.value(slot);
    r.text(name);
    job.drapes.emplace_back(int(slot), std::move(name));
  }
  if(!r.ok)
    return std::nullopt;
  return job;
}

}  // namespace fv::physics::cloth
