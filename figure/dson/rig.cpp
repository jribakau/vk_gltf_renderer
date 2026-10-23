#include <figure/dson/rig.hpp>

#include <unordered_map>

namespace fv::dson {

namespace {

std::vector<JointChannel> jointChannels(const Json* arr, double defaults)
{
  std::vector<JointChannel> out;
  if(!arr || !arr->is_array())
    return out;
  for(const Json& c : *arr)
  {
    const std::string id = getString(&c, "id");
    if(id != "x" && id != "y" && id != "z" && id != "general")
      continue;
    JointChannel ch;
    ch.axis = id;
    // String(c.label ?? c.name ?? c.id)
    ch.label        = getString(&c, "label", getString(&c, "name", id));
    const Json* min = get(&c, "min");
    const Json* max = get(&c, "max");
    ch.min          = min && min->is_number() ? min->get<double>() : -10000;
    ch.max          = max && max->is_number() ? max->get<double>() : 10000;
    const Json* cl  = get(&c, "clamped");
    const Json* lk  = get(&c, "locked");
    ch.clamped      = cl && cl->is_boolean() ? cl->get<bool>() : (cl && cl->is_number() && cl->get<double>() != 0);
    ch.locked       = lk && lk->is_boolean() ? lk->get<bool>() : (lk && lk->is_number() && lk->get<double>() != 0);
    const Json* v   = get(&c, "current_value");
    if(!v || v->is_null())
      v = get(&c, "value");
    ch.value = v && v->is_number() ? v->get<double>() : defaults;
    out.push_back(std::move(ch));
  }
  return out;
}

// The id after '#' in a DSON parent reference, decoded.
std::string fragmentId(const std::string& ref)
{
  const size_t hash = ref.rfind('#');
  return decodeUri(hash == std::string::npos ? ref : ref.substr(hash + 1));
}

}  // namespace

int Rig::byId(const std::string& id) const
{
  for(const auto& [boneId, node] : bones)
    if(boneId == id)
      return node;
  return -1;
}

int Rig::find(const SceneGraph& scene, const std::string& name) const
{
  if(const int b = byId(name); b >= 0)
    return b;
  for(const auto& [boneId, node] : bones)
    if(scene.nodes[node].name == name)
      return node;
  return -1;
}

std::optional<Rig> buildRig(SceneGraph& scene, const Json& nodeLibrary, const std::map<std::string, std::array<double, 3>>* centers)
{
  std::vector<const Json*> defs;
  if(nodeLibrary.is_array())
    for(const Json& n : nodeLibrary)
    {
      const std::string type = getString(&n, "type");
      if(type == "bone" || type == "figure")
        defs.push_back(&n);
    }
  if(defs.empty())
    return std::nullopt;
  Rig                                                    rig;
  std::unordered_map<std::string, std::array<double, 3>> pivots;
  for(const Json* n : defs)
  {
    const std::string id = getString(n, "id");
    auto              joint = std::make_shared<JointData>();
    joint->id               = id;
    joint->label            = getString(n, "label");
    const auto authored     = vec3Channel(get(n, "center_point"));
    std::array<double, 3> c = authored;
    if(centers)
      if(auto it = centers->find(id); it != centers->end())
        c = it->second;
    pivots[id]             = c;
    joint->orientation     = vec3Channel(get(n, "orientation"));
    joint->rotationOrder   = getString(n, "rotation_order", "XYZ");
    joint->center          = c;
    joint->centerAuthored  = authored;
    joint->endPoint        = vec3Channel(get(n, "end_point"));
    joint->rotation        = jointChannels(get(n, "rotation"), 0);
    joint->translation     = jointChannels(get(n, "translation"), 0);
    joint->scale           = jointChannels(get(n, "scale"), 1);
    if(const Json* gs = get(n, "general_scale"); gs && !gs->is_null() && !(gs->is_boolean() && !gs->get<bool>()))
    {
      Json general = *gs;
      if(general.is_object())
        general["id"] = "general";
      const Json list = Json::array({general});
      for(auto& ch : jointChannels(&list, 1))
        joint->scale.push_back(std::move(ch));
    }
    Node bone;
    bone.name  = getString(n, "name", id);
    bone.kind  = NodeKind::eBone;
    bone.joint = joint;
    // a bone of the same id later in the library replaces the earlier one in the map, as Map.set does
    const int node = scene.add(std::move(bone));
    bool      replaced = false;
    for(auto& entry : rig.bones)
      if(entry.first == id)
      {
        entry.second = node;
        replaced     = true;
      }
    if(!replaced)
      rig.bones.emplace_back(id, node);
  }
  for(const Json* n : defs)
  {
    const std::string id     = getString(n, "id");
    const int         b      = rig.byId(id);
    const Json*       parent = get(n, "parent");
    const int         p      = parent && parent->is_string() ? rig.byId(fragmentId(parent->get<std::string>())) : -1;
    const auto&       c      = pivots[id];
    if(p >= 0)
    {
      scene.attach(b, p);
      const auto& pc               = pivots[scene.nodes[p].joint->id];
      scene.nodes[b].position      = {c[0] - pc[0], c[1] - pc[1], c[2] - pc[2]};
    }
    else
    {
      scene.nodes[b].position = {c[0], c[1], c[2]};
      if(rig.root < 0)
        rig.root = b;
    }
  }
  if(rig.root < 0)
    return std::nullopt;
  return rig;
}

std::optional<SkinBinding> skinBinding(const Doc& doc, const Json& skin, const Rig& rig, int vertexCount)
{
  const Json* joints = get(&skin, "joints");
  if(!joints || !joints->is_array() || joints->empty())
    return std::nullopt;
  const int                nv = vertexCount;
  SkinBinding              out;
  std::unordered_map<int, int> boneIndex;
  // influences per cage vertex, gathered as (vertex, bone, weight) and sorted into rows below
  std::vector<int32_t> counts(size_t(nv) + 1, 0);
  struct Entry
  {
    int   v, b;
    float w;
  };
  std::vector<Entry> entries;
  for(const Json& j : *joints)
  {
    const Json*       ref  = get(&j, "node");
    const std::string from = ref && ref->is_string() ? ref->get<std::string>() : getString(&j, "id");
    const int         bone = rig.byId(fragmentId(from));
    if(bone < 0)
      continue;
    auto [it, made] = boneIndex.try_emplace(bone, int(out.order.size()));
    if(made)
      out.order.push_back(bone);
    const int bi  = it->second;
    auto      add = [&](double v, double w) {
      if(v >= 0 && v < nv && w > 0)
      {
        counts[size_t(v) + 1]++;
        entries.push_back({int(v), bi, float(w)});
      }
    };
    if(const Json* nw = get(&j, "node_weights"); nw && get(nw, "values"))
    {
      const auto pairs = doc.rows(get(nw, "values"));
      for(int i = 0; i < pairs->count; i++)
        add(pairs->cell(i, 0), pairs->cell(i, 1));
    }
    else if(const Json* local = get(&j, "local_weights"); local && !local->is_null())
    {
      // TriAx (Genesis 2) fallback: take the strongest axis weight per vertex, in the order first met
      std::vector<std::pair<double, double>>   acc;
      std::unordered_map<double, size_t>       at;
      for(const char* axis : {"x", "y", "z"})
      {
        const Json* a     = get(local, axis);
        const auto  pairs = doc.rows(a ? get(a, "values") : nullptr);
        for(int i = 0; i < pairs->count; i++)
        {
          const double v = pairs->cell(i, 0), w = pairs->cell(i, 1);
          auto [slot, fresh] = at.try_emplace(v, acc.size());
          if(fresh)
            acc.emplace_back(v, w);
          else
            acc[slot->second].second = std::max(acc[slot->second].second, w);
        }
      }
      for(const auto& [v, w] : acc)
        add(v, w);
    }
  }
  if(out.order.empty())
    return std::nullopt;
  for(int v = 0; v < nv; v++)
    counts[v + 1] += counts[v];
  std::vector<int32_t> fill(counts.begin(), counts.end() - 1);
  out.input.bone.assign(entries.size(), 0);
  out.input.weight.assign(entries.size(), 0.f);
  for(const Entry& e : entries)
  {
    const int at          = fill[e.v]++;
    out.input.bone[at]   = e.b;
    out.input.weight[at] = e.w;
  }
  out.input.start = std::move(counts);
  auto root       = boneIndex.find(rig.root);
  out.input.rootBone = root == boneIndex.end() ? 0 : root->second;
  return out;
}

}  // namespace fv::dson
