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

// A number as the cooker parses it, five decimals at most (fv_host_support.cpp fmtNumber).
std::string shortNum(float v)
{
  std::string out = num(v);
  while(out.size() > 1 && out.back() == '0')
    out.pop_back();
  if(!out.empty() && out.back() == '.')
    out.pop_back();
  return out == "-0" ? "0" : out;
}

std::string shortRgb(const std::optional<std::array<float, 3>>& c)
{
  return c ? shortNum((*c)[0]) + "," + shortNum((*c)[1]) + "," + shortNum((*c)[2]) : "-";
}

}  // namespace

std::string irisQuery(const IrisRequest& r)
{
  std::string discs;
  for(const auto& d : r.discs)
    discs += (discs.empty() ? "" : ";") + shortNum(d[0]) + "," + shortNum(d[1]) + "," + shortNum(d[2]);
  return shortRgb(r.iris) + "|" + shortNum(r.strength) + "|" + (r.mode == IrisRequest::Mode::eIris ? "iris" : "eye") + "|" + discs + "|" + shortRgb(r.sclera);
}

std::string recolorQuery(const RecolorRequest& r)
{
  return shortRgb(r.color) + "|" + shortNum(r.strength);
}

std::string requestKey(const TextureRequest& req)
{
  std::string key = req.rel + "|" + (req.srgb ? "1" : "0") + (req.normal ? "n" : "") + "|" + req.alphaRel + "|" + req.channels;
  if(!req.bumpRel.empty())
    key += "|bump:" + req.bumpRel + ":" + num(req.bumpStrength);
  if(req.bumpHeight > 0)
    key += "|bh:" + num(req.bumpHeight);
  if(req.makeup)
  {
    const MakeupRequest& m = *req.makeup;
    key += std::string("|mk:") + (m.baseIsSource ? "1" : "0") + ":" + m.colorRel + ":" + m.weightRel + ":" + num(m.weightFactor) + ":" + rgb(m.tint)
           + ":" + rgb(m.factor);
  }
  if(req.iris)
    key += "|ir:" + irisQuery(*req.iris);
  if(req.recolor)
    key += "|rc:" + recolorQuery(*req.recolor) + ":" + req.recolor->reference;
  for(const std::string& layer : req.layers)
    key += "|ly:" + layer;
  return key;
}

std::vector<std::string> mapLayers(const std::string& map)
{
  std::vector<std::string> out;
  size_t                   from = 0;
  for(size_t at; (at = map.find(kLayerSeparator, from)) != std::string::npos; from = at + 1)
    out.push_back(map.substr(from, at - from));
  out.push_back(map.substr(from));
  return out;
}

}  // namespace fv::mat
