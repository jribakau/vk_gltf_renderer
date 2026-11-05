#include <figure/physics/cloth/clusters.hpp>

#include <algorithm>

namespace fv::physics::cloth {

namespace {

enum Group
{
  eStretchGroup,
  eReachGroup,
  eRestGroup,
};

Group groupOf(DistanceKind kind)
{
  return kind == DistanceKind::eStretch ? eStretchGroup : kind == DistanceKind::eReach ? eReachGroup : eRestGroup;
}

// Greedy colours for `rows` (indices into model.edges) of one group, in their order: no two of a colour
// share an end that is written (`writes(v)`; `mask` per vertex index `at(v)`). Returns each row's colour,
// or an empty list when a vertex has more rows than 63 colours.
template <class At, class Writes>
std::vector<int> colour(const Model& model, const std::vector<int>& rows, std::vector<uint64_t>& mask, const At& at, const Writes& writes, int& colours)
{
  std::vector<int> out(rows.size(), 0);
  colours = 0;
  for(size_t k = 0; k < rows.size(); k++)
  {
    const Distance& e  = model.edges[size_t(rows[k])];
    const bool      fa = writes(e.a), fb = writes(e.b);
    const uint64_t  taken = (fa ? mask[size_t(at(e.a))] : 0) | (fb ? mask[size_t(at(e.b))] : 0);
    int             c     = 0;
    while(c < 63 && (taken >> c) & 1)
      c++;
    if(c >= 63)
      return {};
    out[k] = c;
    if(fa)
      mask[size_t(at(e.a))] |= uint64_t(1) << c;
    if(fb)
      mask[size_t(at(e.b))] |= uint64_t(1) << c;
    colours = std::max(colours, c + 1);
  }
  for(size_t k = 0; k < rows.size(); k++)
  {
    const Distance& e = model.edges[size_t(rows[k])];
    if(writes(e.a))
      mask[size_t(at(e.a))] = 0;
    if(writes(e.b))
      mask[size_t(at(e.b))] = 0;
  }
  return out;
}

}  // namespace

Clusters buildClusters(const Model& model, int size, int mostLocal, bool halo)
{
  Clusters     out;
  const size_t nv   = size_t(model.nv);
  const auto   free = [&](int v) { return model.inv[size_t(v)] > 0; };
  // each free vertex's pinned neighbours: the ghosts it brings into its cluster
  std::vector<std::vector<int>> pinnedNear(nv);
  for(const Distance& e : model.edges)
  {
    if(free(e.a) && !free(e.b))
      pinnedNear[size_t(e.a)].push_back(e.b);
    else if(free(e.b) && !free(e.a))
      pinnedNear[size_t(e.b)].push_back(e.a);
  }
  // free neighbours along the surface (stretch and shear rows), which a cluster grows over
  std::vector<std::vector<int>> freeNear(nv);
  for(const Distance& e : model.edges)
    if((e.kind == DistanceKind::eStretch || e.kind == DistanceKind::eShear) && free(e.a) && free(e.b))
    {
      freeNear[size_t(e.a)].push_back(e.b);
      freeNear[size_t(e.b)].push_back(e.a);
    }
  // the clusters: patches grown breadth first over the surface, each closed at `size` owned or before its
  // ghosts would take it past `mostLocal`, the next grown from where the last stopped. Compact patches cut
  // fewer rows than runs of the vertex order: runs of 1024 cut a quarter of the Kai Dress's rows. A piece
  // the patch has all of (a card of a hair) leaves it open for the next, so many small ones share a
  // workgroup rather than each taking one
  std::vector<int>  clusterOf(nv, -1);
  std::vector<int>  ghostOf(nv, -1);  // the cluster a pinned vertex is a ghost of, while that cluster is built
  std::vector<int>  ghosts;
  std::vector<int>  owned;
  const auto        close = [&] {
    if(owned.empty())
      return;
    const int c = out.count();
    // in index order, so a workgroup reads and writes them in runs
    std::sort(owned.begin(), owned.end());
    out.verts.insert(out.verts.end(), owned.begin(), owned.end());
    out.verts.insert(out.verts.end(), ghosts.begin(), ghosts.end());
    out.vertStart.push_back(int(out.verts.size()));
    out.owned.push_back(int(owned.size()));
    out.mostLocal = std::max(out.mostLocal, int(owned.size() + ghosts.size()));
    for(int v : owned)
      clusterOf[size_t(v)] = c;
    owned.clear();
    ghosts.clear();
  };
  std::vector<char> queued(nv, 0);
  std::vector<int>  queue, frontier;
  size_t            nextSeed = 0;
  for(;;)
  {
    // the seed: where the last patch stopped, else the first free vertex not yet taken
    int seed = -1;
    while(!frontier.empty() && seed < 0)
    {
      const int v = frontier.back();
      frontier.pop_back();
      if(clusterOf[size_t(v)] < 0 && !queued[size_t(v)])
        seed = v;
    }
    while(seed < 0 && nextSeed < model.freeVerts.size())
    {
      const int v = model.freeVerts[nextSeed++];
      if(clusterOf[size_t(v)] < 0)
        seed = v;
    }
    if(seed < 0)
      break;
    queue.assign(1, seed);
    queued[size_t(seed)] = 1;
    size_t head = 0;
    for(; head < queue.size(); head++)
    {
      const int v     = queue[head];
      int       fresh = 0;
      for(int g : pinnedNear[size_t(v)])
        if(ghostOf[size_t(g)] != out.count())
          fresh++;
      if(!owned.empty() && (int(owned.size()) >= size || int(owned.size() + ghosts.size()) + 1 + fresh > mostLocal))
        break;
      owned.push_back(v);
      // taken now: a cluster left open for the next piece mustn't take it again
      clusterOf[size_t(v)] = out.count();
      for(int g : pinnedNear[size_t(v)])
        if(ghostOf[size_t(g)] != out.count())
        {
          ghostOf[size_t(g)] = out.count();
          ghosts.push_back(g);
        }
      for(int u : freeNear[size_t(v)])
        if(clusterOf[size_t(u)] < 0 && !queued[size_t(u)])
        {
          queued[size_t(u)] = 1;
          queue.push_back(u);
        }
    }
    // what was queued and not taken goes back, and the next patch grows from it: the first of it on top
    const bool full = head < queue.size();
    for(size_t k = queue.size(); k-- > head;)
      frontier.push_back(queue[k]);
    for(int v : queue)
      queued[size_t(v)] = 0;
    if(full)
      close();
  }
  close();

  // each row to the cluster of its free ends, or to the crossing rows when they lie in two
  std::vector<std::vector<int>> rowsOf(size_t(out.count()) * 3);
  std::vector<std::vector<int>> haloOf(size_t(out.count()));
  std::vector<int>              crossingRows[3];
  for(size_t r = 0; r < model.edges.size(); r++)
  {
    const Distance& e = model.edges[r];
    if(!free(e.a) && !free(e.b))
      continue;
    const int ca = free(e.a) ? clusterOf[size_t(e.a)] : -1, cb = free(e.b) ? clusterOf[size_t(e.b)] : -1;
    const int g  = groupOf(e.kind);
    if(ca >= 0 && cb >= 0 && ca != cb)
    {
      if(!halo)
      {
        crossingRows[g].push_back(int(r));
        continue;
      }
      // both clusters hold it, each with the other's end as a halo vertex
      out.haloRows++;
      rowsOf[size_t(ca) * 3 + size_t(g)].push_back(int(r));
      rowsOf[size_t(cb) * 3 + size_t(g)].push_back(int(r));
      haloOf[size_t(ca)].push_back(e.b);
      haloOf[size_t(cb)].push_back(e.a);
    }
    else
      rowsOf[size_t(ca >= 0 ? ca : cb) * 3 + size_t(g)].push_back(int(r));
  }
  if(halo)
  {
    // each cluster's halo vertices after its ghosts, once each
    std::vector<int> verts, vertStart{0};
    std::vector<int> seen(nv, -1);
    out.mostLocal = 0;
    for(int c = 0; c < out.count(); c++)
    {
      for(int k = out.vertStart[size_t(c)]; k < out.vertStart[size_t(c) + 1]; k++)
      {
        verts.push_back(out.verts[size_t(k)]);
        seen[size_t(out.verts[size_t(k)])] = c;
      }
      for(int v : haloOf[size_t(c)])
        if(seen[size_t(v)] != c)
        {
          seen[size_t(v)] = c;
          verts.push_back(v);
        }
      vertStart.push_back(int(verts.size()));
      out.mostLocal = std::max(out.mostLocal, vertStart.back() - vertStart[vertStart.size() - 2]);
    }
    out.verts     = std::move(verts);
    out.vertStart = std::move(vertStart);
  }

  // colour each cluster's rows, group by group, over its local indices
  std::vector<int>      localOf(nv, -1);
  std::vector<uint64_t> mask(size_t(std::max(out.mostLocal, 1)), 0);
  struct Coloured
  {
    std::vector<int> colourOf[3];
    int              colours[3] = {0, 0, 0};
  };
  std::vector<Coloured> coloured(size_t(out.count()));
  for(int c = 0; c < out.count(); c++)
  {
    for(int k = out.vertStart[size_t(c)]; k < out.vertStart[size_t(c) + 1]; k++)
      localOf[size_t(out.verts[size_t(k)])] = k - out.vertStart[size_t(c)];
    // a cluster writes its own vertices alone: ghosts and halo vertices it only reads
    const int  owned  = out.owned[size_t(c)];
    const auto writes = [&](int v) { return free(v) && localOf[size_t(v)] < owned; };
    for(int g = 0; g < 3; g++)
    {
      coloured[size_t(c)].colourOf[g] = colour(model, rowsOf[size_t(c) * 3 + size_t(g)], mask, [&](int v) { return localOf[size_t(v)]; }, writes,
                                               coloured[size_t(c)].colours[g]);
      if(coloured[size_t(c)].colourOf[g].empty() && !rowsOf[size_t(c) * 3 + size_t(g)].empty())
        return {};
    }
    out.stretchColours = std::max(out.stretchColours, coloured[size_t(c)].colours[eStretchGroup]);
    out.reachColours   = std::max(out.reachColours, coloured[size_t(c)].colours[eReachGroup]);
    out.restColours    = std::max(out.restColours, coloured[size_t(c)].colours[eRestGroup]);
    for(int k = out.vertStart[size_t(c)]; k < out.vertStart[size_t(c) + 1]; k++)
      localOf[size_t(out.verts[size_t(k)])] = -1;
  }
  // the rows in colour order, every cluster with the same colours in each group
  const int perGroup[3] = {out.stretchColours, out.reachColours, out.restColours};
  out.colourStart.reserve(size_t(out.count()) * size_t(out.colours()) + 1);
  for(int c = 0; c < out.count(); c++)
  {
    for(int k = out.vertStart[size_t(c)]; k < out.vertStart[size_t(c) + 1]; k++)
      localOf[size_t(out.verts[size_t(k)])] = k - out.vertStart[size_t(c)];
    for(int g = 0; g < 3; g++)
      for(int k = 0; k < perGroup[g]; k++)
      {
        out.colourStart.push_back(int(out.rows.size()));
        const std::vector<int>& rows = rowsOf[size_t(c) * 3 + size_t(g)];
        for(size_t r = 0; r < rows.size(); r++)
          if(coloured[size_t(c)].colourOf[g][r] == k)
          {
            const Distance& e = model.edges[size_t(rows[r])];
            out.rows.push_back({uint16_t(localOf[size_t(e.a)]), uint16_t(localOf[size_t(e.b)]), rows[r]});
          }
      }
    for(int k = out.vertStart[size_t(c)]; k < out.vertStart[size_t(c) + 1]; k++)
      localOf[size_t(out.verts[size_t(k)])] = -1;
  }
  out.colourStart.push_back(int(out.rows.size()));

  // the tethered vertices, by cluster
  {
    std::vector<std::vector<std::pair<uint16_t, int>>> softOf(size_t(out.count()));
    for(size_t s = 0; s < model.softVerts.size(); s++)
    {
      const int v = model.softVerts[s];
      const int c = clusterOf[size_t(v)];
      if(c < 0)
        continue;
      const auto first = out.verts.begin() + out.vertStart[size_t(c)];
      const auto last  = first + out.owned[size_t(c)];
      softOf[size_t(c)].emplace_back(uint16_t(std::find(first, last, v) - first), int(s));
    }
    for(const auto& list : softOf)
    {
      out.soft.insert(out.soft.end(), list.begin(), list.end());
      out.softStart.push_back(int(out.soft.size()));
    }
  }

  // the crossing rows, coloured over the model's vertices
  std::vector<uint64_t> globalMask(nv, 0);
  int*                  counts[3] = {&out.crossingStretch, &out.crossingReach, &out.crossingRest};
  for(int g = 0; g < 3; g++)
  {
    const std::vector<int> colourOf = colour(model, crossingRows[g], globalMask, [](int v) { return v; }, free, *counts[g]);
    if(colourOf.empty() && !crossingRows[g].empty())
      return {};
    for(int k = 0; k < *counts[g]; k++)
    {
      for(size_t r = 0; r < crossingRows[g].size(); r++)
        if(colourOf[r] == k)
          out.crossing.push_back(crossingRows[g][r]);
      out.crossingStart.push_back(int(out.crossing.size()));
    }
  }
  return out;
}

}  // namespace fv::physics::cloth
