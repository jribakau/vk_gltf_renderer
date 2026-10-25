#include <figure/dson/channels.hpp>

#include <algorithm>
#include <cctype>

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

void Channels::erase(const std::string& key)
{
  if(m_map.erase(key))
    m_order.erase(std::find(m_order.begin(), m_order.end(), key));
}

void Channels::set(const std::string& key, std::shared_ptr<Channel> c)
{
  auto [it, made] = m_map.insert_or_assign(key, std::move(c));
  if(made)
    m_order.push_back(key);
}

}  // namespace fv::dson
