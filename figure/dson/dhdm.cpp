#include <figure/dson/dhdm.hpp>

#include <cstring>
#include <string>

namespace fv::dson {

namespace {

struct Reader
{
  std::string_view bytes;
  size_t           at = 0;

  void need(size_t n, const char* what) const
  {
    if(bytes.size() - at < n)
      throw std::runtime_error(std::string("HD morph file ends inside ") + what);
  }
  template <class T>
  T read(const char* what)
  {
    need(sizeof(T), what);
    T v;
    std::memcpy(&v, bytes.data() + at, sizeof(T));
    at += sizeof(T);
    return v;
  }
};

constexpr uint32_t kMagic = 0xd0d0d0d0u;
constexpr uint32_t kOne   = 0x3f800000u;  // 1.0f

}  // namespace

Dhdm parseDhdm(std::string_view bytes)
{
  Reader r{bytes};
  if(r.read<uint32_t>("its header") != kMagic)
    throw std::runtime_error("not an HD morph file");
  const uint32_t levels = r.read<uint32_t>("its header");
  if(r.read<uint32_t>("its header") != kOne || r.read<uint32_t>("its header") != levels)
    throw std::runtime_error("not an HD morph file");
  // a 16-bit index holds a corner and 2 bits a level: seven levels at most
  if(levels > 7)
    throw std::runtime_error("HD morph file states " + std::to_string(levels) + " levels");
  Dhdm out;
  out.levels.resize(levels);
  for(uint32_t level = 1; level <= levels; level++)
  {
    const uint32_t faces = r.read<uint32_t>("a level header");
    if(r.read<uint32_t>("a level header") != level)
      throw std::runtime_error("HD morph file's levels are out of order");
    const uint32_t count = r.read<uint32_t>("a level header");
    const uint32_t size  = r.read<uint32_t>("a level header");
    if(level > 1 && faces != out.faceCount)
      throw std::runtime_error("HD morph file's levels state different cages");
    out.faceCount = faces;
    r.need(size, "a level");
    Reader l{bytes.substr(r.at, size)};
    r.at += size;
    std::vector<subdiv::FaceEdit>& edits = out.levels[level - 1];
    edits.reserve(count);
    const int shift = 16 - 2 * int(level);
    while(edits.size() < count)
    {
      const uint32_t face = l.read<uint32_t>("a face header");
      const uint32_t n    = l.read<uint32_t>("a face header");
      for(uint32_t i = 0; i < n && edits.size() < count; i++)
      {
        subdiv::FaceEdit e;
        e.face    = face;
        e.offset[0] = l.read<float>("an edit");
        uint32_t index = 0;
        if(level < 4)
        {
          index = uint32_t(l.read<uint8_t>("an edit")) << 8;
          l.read<uint8_t>("an edit");
        }
        else
        {
          l.read<uint8_t>("an edit");
          index = l.read<uint16_t>("an edit");
          l.read<uint8_t>("an edit");
        }
        e.offset[1] = l.read<float>("an edit");
        e.offset[2] = l.read<float>("an edit");
        e.sub       = index >> shift;
        e.corner    = uint8_t((index >> (shift - 2)) & 3);
        edits.push_back(e);
      }
    }
  }
  return out;
}

}  // namespace fv::dson
