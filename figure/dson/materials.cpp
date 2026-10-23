#include <figure/dson/materials.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>

namespace fv::dson {

std::string norm(std::string_view s)
{
  std::string out;
  out.reserve(s.size());
  for(char c : s)
  {
    const char l = char(std::tolower(static_cast<unsigned char>(c)));
    if((l >= 'a' && l <= 'z') || (l >= '0' && l <= '9'))
      out += l;
  }
  return out;
}

const Channel* Channels::get(const std::string& normName) const
{
  auto it = m_map.find(normName);
  return it == m_map.end() ? nullptr : it->second.get();
}

std::shared_ptr<Channel> Channels::shared(const std::string& key) const
{
  auto it = m_map.find(key);
  return it == m_map.end() ? nullptr : it->second;
}

const Channel* Channels::find(std::initializer_list<const char*> names) const
{
  for(const char* n : names)
    if(const Channel* c = get(norm(n)))
      return c;
  return nullptr;
}

void Channels::set(const std::string& key, std::shared_ptr<Channel> c)
{
  auto [it, made] = m_map.insert_or_assign(key, std::move(c));
  if(made)
    m_order.push_back(key);
}

void collectChannels(const Json* def, Channels& into, const std::unordered_map<std::string, std::string>& images, const std::string& file)
{
  if(!def || !def->is_object())
    return;
  auto put = [&](const Json& c) {
    const Json* inner = get(&c, "channel");
    const Json& ch    = inner && !inner->is_null() ? *inner : c;
    std::vector<std::string> own;
    for(const char* k : {"id", "label", "name"})
    {
      const std::string s = getString(&ch, k);
      if(!s.empty())
        own.push_back(norm(s));
    }
    if(own.empty())
      return;
    std::shared_ptr<Channel> prev;
    for(const auto& k : own)
      if((prev = into.shared(k)))
        break;
    // an override also replaces the names the earlier channel was stored under: the figure's Iray
    // "diffuse"/"Base Color" channel is overridden by a character's "diffuse"/"Diffuse Color"
    std::vector<std::string> keys = own;
    if(prev)
      for(const auto& k : prev->keys)
        if(into.shared(k) == prev)
          keys.push_back(k);
    {
      std::vector<std::string> unique;
      for(const auto& k : keys)
        if(std::find(unique.begin(), unique.end(), k) == unique.end())
          unique.push_back(k);
      keys = std::move(unique);
    }
    auto entry = std::make_shared<Channel>();
    if(const Json* v = get(&ch, "current_value"); v && !v->is_null())
      entry->value = *v;
    else if(const Json* v2 = get(&ch, "value"); v2 && !v2->is_null())
      entry->value = *v2;
    else if(prev)
      entry->value = prev->value;
    entry->image = prev ? prev->image : std::nullopt;
    if(ch.contains("image_file") || ch.contains("image"))
    {
      entry->image.reset();
      const std::string imageFile = getString(&ch, "image_file");
      const std::string image     = getString(&ch, "image");
      if(!imageFile.empty())
        entry->image = imageFile;
      else if(!image.empty())
      {
        auto ref = parseRef(image, file);
        if(ref)
          if(auto it = images.find(ref->id); it != images.end())
            entry->image = it->second;
      }
    }
    entry->keys = keys;
    for(const auto& k : keys)
      into.set(k, entry);
  };
  for(auto it = def->begin(); it != def->end(); ++it)
  {
    const Json& v = it.value();
    if(v.is_object() && v.contains("channel"))
      put(v);
    else if(it.key() == "extra" && v.is_array())
      for(const Json& ex : v)
        if(const Json* chans = get(&ex, "channels"); chans && chans->is_array())
          for(const Json& c : *chans)
            put(c);
  }
}

void collectImages(const Json& doc, std::unordered_map<std::string, std::string>& into)
{
  const Json* lib = get(&doc, "image_library");
  if(!lib || !lib->is_array())
    return;
  for(const Json& img : *lib)
  {
    const Json* layers = get(&img, "map");
    if(!layers || !layers->is_array())
      continue;
    for(const Json& l : *layers)
    {
      const Json* active = get(&l, "active");
      if(active && active->is_boolean() && !active->get<bool>())
        continue;
      const std::string url = getString(&l, "url");
      if(url.empty())
        continue;
      const std::string id = getString(&img, "id");
      if(!id.empty())
        into[id] = url;
      break;
    }
  }
}

std::string materialKey(const std::string& name, const Channels& ch)
{
  std::string key = name;
  for(const auto& k : ch.order())
  {
    const Channel* c = ch.get(k);
    key += "\n" + k + "=" + (c->value.is_null() ? std::string() : c->value.dump()) + "|" + c->image.value_or("");
  }
  return key;
}

float srgbToLinear(double c)
{
  return float(c < 0.04045 ? c * 0.0773993808 : std::pow(c * 0.9478672986 + 0.0521327014, 2.4));
}

namespace {

double num(const Channel* c, double d)
{
  return c && c->value.is_number() ? c->value.get<double>() : d;
}

}  // namespace

Material buildMaterial(const std::string& name, const Channels& ch)
{
  Material m;
  m.name = name;
  if(const Channel* base = ch.find({"Base Color", "diffuse", "Diffuse Color"}))
  {
    // DSON colour channels hold sRGB values, as shown in the DAZ Studio colour pickers
    if(base->value.is_array() && base->value.size() >= 3 && base->value[0].is_number())
      for(int i = 0; i < 3; i++)
        m.baseColor[i] = srgbToLinear(base->value[i].get<double>());
    if(base->image)
      m.baseColorMap = *base->image;
  }
  if(const Channel* cut = ch.find({"Cutout Opacity", "transparency", "opacity"}))
  {
    m.opacity = float(num(cut, 1));
    if(cut->image)
      m.opacityMap = *cut->image;
  }
  if(const Channel* rough = ch.find({"Glossy Roughness", "Specular Lobe 1 Roughness", "roughness"}))
    m.roughness = float(num(rough, 0.5));
  for(const auto& k : ch.order())
  {
    const Channel* c = ch.get(k);
    m.channels[k]    = {c->value.is_null() ? std::string() : c->value.dump(), c->image.value_or("")};
  }
  return m;
}

}  // namespace fv::dson
