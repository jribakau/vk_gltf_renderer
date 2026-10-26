#include <figure/wardrobe/follow.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

#include <figure/io/timing.hpp>
#include <figure/jobs/pool.hpp>

namespace fv::wardrobe {

std::optional<Rigidity> parseRigidity(const dson::Doc& doc, const dson::Json& geo, int vertexCount)
{
  const dson::Json* r = dson::get(&geo, "rigidity");
  if(!r)
    return std::nullopt;
  const dson::Json* groups = dson::get(r, "groups");
  if(!groups || !groups->is_array())
    return std::nullopt;
  const auto listOf = [&](const dson::Json* g, const char* key) {
    std::vector<uint32_t> out;
    const dson::Json*     values = dson::get(dson::get(g, key), "values");
    if(values && values->is_array())
      for(const dson::Json& v : *values)
        if(v.is_number() && v.get<double>() >= 0 && v.get<double>() < vertexCount)
          out.push_back(uint32_t(v.get<double>()));
    return out;
  };
  Rigidity out;
  for(const dson::Json& g : *groups)
  {
    Rigidity::Group group{listOf(&g, "reference_vertices"), listOf(&g, "mask_vertices")};
    if(!group.refs.empty() && !group.mask.empty())
      out.groups.push_back(std::move(group));
  }
  if(out.groups.empty())
    return std::nullopt;
  out.weight.assign(size_t(vertexCount), 0.f);
  const auto weights = doc.rows(dson::get(dson::get(r, "weights"), "values"));
  for(int i = 0; i < weights->count; i++)
  {
    const double v = weights->cell(i, 0);
    if(v >= 0 && v < vertexCount)
      out.weight[size_t(v)] = float(weights->cell(i, 1));
  }
  return out;
}

std::optional<Graft> parseGraft(const dson::Doc& doc, const dson::Json& geo)
{
  const dson::Json* g = dson::get(&geo, "graft");
  if(!g)
    return std::nullopt;
  Graft      out;
  const auto table = doc.rows(dson::get(dson::get(g, "vertex_pairs"), "values"));
  for(int i = 0; i < table->count; i++)
  {
    out.pairs.push_back(int32_t(table->cell(i, 0)));
    out.pairs.push_back(int32_t(table->cell(i, 1)));
  }
  const dson::Json* hidden = dson::get(dson::get(g, "hidden_polys"), "values");
  if(hidden && hidden->is_array())
    for(const dson::Json& v : *hidden)
      if(v.is_number())
        out.hidden.push_back(int32_t(v.get<double>()));
  if(out.pairs.empty() && out.hidden.empty())
    return std::nullopt;
  return out;
}

dson::TargetPtr projectTarget(const SurfaceBinding& binding, const Rigidity* rigidity, const mesh::BuiltMesh& body, const mesh::BuiltMesh& item,
                              const mesh::RenderTarget& t, std::vector<float>& scratch)
{
  const auto& indices = t.cageIndices;
  const auto& deltas  = t.cageDeltas;
  if(indices.empty())
    return nullptr;
  if(scratch.size() != size_t(body.dsonVertexCount) * 3)
    scratch.assign(size_t(body.dsonVertexCount) * 3, 0.f);
  bool any = false;
  for(size_t i = 0; i < indices.size(); i++)
  {
    const size_t d = indices[i];
    if(d >= binding.used.size())
      continue;
    if(binding.used[d])
      any = true;
    scratch[d * 3]     = deltas[i * 3];
    scratch[d * 3 + 1] = deltas[i * 3 + 1];
    scratch[d * 3 + 2] = deltas[i * 3 + 2];
  }
  const auto clear = [&] {
    for(uint32_t d : indices)
      if(d < binding.used.size())
        scratch[size_t(d) * 3] = scratch[size_t(d) * 3 + 1] = scratch[size_t(d) * 3 + 2] = 0.f;
  };
  if(!any)
  {
    clear();
    return nullptr;
  }
  std::optional<fv::timing::Accum> step;
  step.emplace("project: item verts");
  // a long item (a hair's strands) is worked a chunk at a time side by side; each vertex is its own
  constexpr size_t   kChunk = 32768;
  const size_t       nv = size_t(item.dsonVertexCount), chunks = (nv + kChunk - 1) / kChunk;
  std::vector<float> out(nv * 3);
  jobs::parallelFor(nv, kChunk, [&](size_t begin, size_t end) {
    for(size_t v = begin; v < end; v++)
    {
      const size_t k = v * 3;
      const size_t a = binding.verts[k] * 3ull, b = binding.verts[k + 1] * 3ull, c = binding.verts[k + 2] * 3ull;
      const float  wa = binding.weights[k], wb = binding.weights[k + 1], wc = binding.weights[k + 2];
      for(int i = 0; i < 3; i++)
        out[k + i] = scratch[a + i] * wa + scratch[b + i] * wb + scratch[c + i] * wc;
    }
  });
  clear();
  if(rigidity)
    for(const Rigidity::Group& g : rigidity->groups)
    {
      float x = 0, y = 0, z = 0;
      for(uint32_t r : g.refs)
      {
        x += out[r * 3ull];
        y += out[r * 3ull + 1];
        z += out[r * 3ull + 2];
      }
      x /= float(g.refs.size());
      y /= float(g.refs.size());
      z /= float(g.refs.size());
      for(uint32_t m : g.mask)
      {
        const float w = rigidity->weight[m];
        out[m * 3ull] += (x - out[m * 3ull]) * w;
        out[m * 3ull + 1] += (y - out[m * 3ull + 1]) * w;
        out[m * 3ull + 2] += (z - out[m * 3ull + 2]) * w;
      }
    }
  // the vertices it moves, in order: counted a chunk at a time, then written where each chunk's go
  const auto          moves = [&](size_t v) { return !(std::abs(out[v * 3]) + std::abs(out[v * 3 + 1]) + std::abs(out[v * 3 + 2]) < 1e-5f); };
  std::vector<size_t> at(chunks + 1, 0);
  jobs::parallelFor(nv, kChunk, [&](size_t begin, size_t end) {
    for(size_t v = begin; v < end; v++)
      at[begin / kChunk + 1] += moves(v);
  });
  std::partial_sum(at.begin(), at.end(), at.begin());
  if(!at[chunks])
    return nullptr;
  std::vector<uint32_t> cageIndices(at[chunks]);
  std::vector<float>    cageDeltas(at[chunks] * 3);
  jobs::parallelFor(nv, kChunk, [&](size_t begin, size_t end) {
    size_t o = at[begin / kChunk];
    for(size_t v = begin; v < end; v++)
      if(moves(v))
      {
        cageIndices[o] = uint32_t(v);
        std::copy_n(&out[v * 3], 3, &cageDeltas[o * 3]);
        o++;
      }
  });
  step.emplace("project: refine");
  auto target = std::make_shared<mesh::RenderTarget>(mesh::target(item, std::move(cageIndices), std::move(cageDeltas)));
  if(target->indices.empty())
    return nullptr;
  return target;
}

}  // namespace fv::wardrobe
