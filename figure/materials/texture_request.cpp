#include <figure/materials/texture_request.hpp>

#include <cstdio>

namespace fv::mat {

namespace {

std::string num(float v)
{
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.5f", v);
  return buf;
}

std::string rgb(const std::array<float, 3>& c)
{
  return num(c[0]) + "," + num(c[1]) + "," + num(c[2]);
}

}  // namespace

std::string requestKey(const TextureRequest& req)
{
  std::string key = req.rel + "|" + (req.srgb ? "1" : "0") + (req.normal ? "n" : "") + "|" + req.alphaRel + "|" + req.channels;
  if(!req.bumpRel.empty())
    key += "|bump:" + req.bumpRel + ":" + num(req.bumpStrength);
  if(req.makeup)
  {
    const MakeupRequest& m = *req.makeup;
    key += std::string("|mk:") + (m.baseIsSource ? "1" : "0") + ":" + m.colorRel + ":" + m.weightRel + ":" + num(m.weightFactor) + ":" + rgb(m.tint)
           + ":" + rgb(m.factor);
  }
  return key;
}

}  // namespace fv::mat
