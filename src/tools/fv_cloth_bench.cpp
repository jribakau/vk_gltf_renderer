// The GPU cloth bench (docs/physics-plan.md step 13a): a cloth job, as fv_parity's dforce step writes
// it ("job"), settled on the CPU island, on the GPU solver a dispatch a colour (today's) and on its
// cluster solve, each timed, its drapes judged by the numbers fv_parity reports for them and set
// against the CPU's vertex by vertex. Needs a Vulkan device and nothing of the app.
//
//   fv_cloth_bench <job> [out.json] [--solvers cpu,gpu,cluster,coloured] [--cluster 1024[,512..]] [--local 2048]
//                  [--analyse] [--grid] [--self] [--trace]
//
// "cluster" has both clusters of a row that crosses them hold it, each moving its own end; "coloured"
// runs those rows after the clusters, a dispatch a colour of them.
//
// --analyse prints how each cluster size cuts the model (clusters, colours, the rows that cross) and
// runs nothing. Every number goes to out.json too.
//
// --grid needs no GPU: the colliders swept through the job's transition, frame by frame as the GPU solver
// sweeps them, and the collision grid made from each frame (figure/physics/collide/sweep_grid.hpp) by the
// builder and by its reference, both timed, and whether they agree.
//
// Each GPU run settles the job twice on one simulation, the second time through restart (physics-plan
// step 13b: it keeps its model, buffers and kernels): "restart" says how long that took and whether its
// drape is the first's to the bit. Then once more with the body dented (its rest cage moved in, as a
// contact pass does), set against a new simulation of that job: "dented" says whether the two agree.

// --self needs no GPU: for each simulated mesh, its own faces that lie under or over its vertices at the
// start (the rest shape), beyond the two rings a vertex never meets, within the gap and within twice it,
// how many such faces the worst vertex has, and the deepest; with the mesh's edge length and offsets. For
// a garment that tears when it meets its own faces: whether it is folded onto itself as made. --trace runs
// the GPU settle a frame at a time and prints the same after each frame (physics-plan step 13d: the Kai
// Dress crumples onto itself from its sixth frame, 5,000 vertices within the gap of an own face).
// the allocator's implementation, as renderer.cpp compiles it for the renderer
#define VMA_IMPLEMENTATION
#include <nvvk/resource_allocator.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <figure/dson/store.hpp>
#include <figure/jobs/pool.hpp>
#include <figure/physics/cloth/clusters.hpp>
#include <figure/physics/cloth/island.hpp>
#include <figure/physics/cloth/job.hpp>
#include <figure/physics/cloth/report.hpp>
#include <figure/physics/collide/sweep_grid.hpp>
#include <figure/physics/sweep.hpp>
#include <nvapp/application.hpp>
#include <nvutils/logger.hpp>
#include <glm/glm.hpp>
#include <nvvk/context.hpp>
#include <nvvk/resource_allocator.hpp>
#include <volk.h>

#include "fv_physics_gpu.hpp"

using fv::dson::Json;
namespace cloth = fv::physics::cloth;

namespace {

double since(std::chrono::steady_clock::time_point t)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

std::vector<std::string> split(const std::string& s)
{
  std::vector<std::string> out;
  std::stringstream        in(s);
  for(std::string part; std::getline(in, part, ',');)
    if(!part.empty())
      out.push_back(part);
  return out;
}

double r4(double v)
{
  return std::round(v * 1e4) / 1e4;
}

// How a cluster size cuts the model.
Json analyse(const cloth::Model& model, int size, int local)
{
  const auto   t0 = std::chrono::steady_clock::now();
  const auto   c  = cloth::buildClusters(model, size, local);
  const double ms = since(t0);
  if(c.count() == 0)
    return {{"size", size}, {"failed", true}};
  size_t inside[4] = {0, 0, 0, 0}, crossing[4] = {0, 0, 0, 0};
  for(const auto& row : c.rows)
    inside[size_t(model.edges[size_t(row.edge)].kind)]++;
  for(int e : c.crossing)
    crossing[size_t(model.edges[size_t(e)].kind)]++;
  size_t ghosts = 0;
  for(int k = 0; k < c.count(); k++)
    ghosts += size_t(c.vertStart[size_t(k) + 1] - c.vertStart[size_t(k)] - c.owned[size_t(k)]);
  const char* kinds[4] = {"stretch", "shear", "bend", "reach"};
  Json        in = Json::object(), across = Json::object();
  for(int k = 0; k < 4; k++)
  {
    in[kinds[k]]     = inside[k];
    across[kinds[k]] = crossing[k];
  }
  return {{"size", size},
          {"clusters", c.count()},
          {"mostLocal", c.mostLocal},
          {"ghosts", ghosts},
          {"colours", {{"stretch", c.stretchColours}, {"reach", c.reachColours}, {"rest", c.restColours}}},
          {"rowsInClusters", in},
          {"rowsCrossing", across},
          {"crossingColours", {{"stretch", c.crossingStretch}, {"reach", c.crossingReach}, {"rest", c.crossingRest}}},
          {"buildMs", r4(ms)}};
}

// A drape's numbers, and how far it lies from the CPU's.
Json judge(const cloth::Job& job, const cloth::Model& model, const std::vector<std::vector<float>>& out, const std::vector<std::vector<float>>* cpu)
{
  Json drapes = Json::array();
  for(const auto& [slot, name] : job.drapes)
  {
    const auto& mesh = job.surfaces[size_t(slot)];
    const auto  d    = cloth::drapeNumbers(model, size_t(slot), mesh, out[size_t(slot)], job.surfaces[0], out[0]);
    Json        j    = {{"mesh", name},
                        {"hemDropCm", r4(d.hemDrop)},
                        {"stretchP99", r4(d.stretchP99)},
                        {"stretchMax", r4(d.stretchMax)},
                        {"stretchOver1.2", d.stretchOver12},
                        {"insideBody", d.inside}};
    // the drape's top, where a dress hangs from the shoulders
    float top = -1e9f;
    for(size_t v = 1; v < out[size_t(slot)].size(); v += 3)
      top = std::max(top, out[size_t(slot)][v]);
    j["topY"] = r4(top);
    // through another garment: its vertices on the other side of each other drape than they were made
    for(const auto& [other, otherName] : job.drapes)
      if(other != slot)
        j["wrongSide"][otherName] = cloth::wrongSide(mesh, out[size_t(slot)], job.surfaces[size_t(other)], out[size_t(other)]);
    // its bits, so two runs can be told the same or not
    uint64_t h = 1469598103934665603ull;
    for(const float f : out[size_t(slot)])
    {
      uint32_t u;
      std::memcpy(&u, &f, sizeof u);
      for(int k = 0; k < 4; k++)
      {
        h ^= (u >> (8 * k)) & 0xFF;
        h *= 1099511628211ull;
      }
    }
    char bits[17];
    std::snprintf(bits, sizeof bits, "%016llx", static_cast<unsigned long long>(h));
    j["bits"] = bits;
    if(cpu)
    {
      std::vector<double> dist;
      const auto&         a = (*cpu)[size_t(slot)];
      const auto&         b = out[size_t(slot)];
      for(size_t v = 0; v + 2 < a.size(); v += 3)
        dist.push_back(std::hypot(double(a[v]) - b[v], double(a[v + 1]) - b[v + 1], double(a[v + 2]) - b[v + 2]));
      std::sort(dist.begin(), dist.end());
      double sum = 0;
      for(double x : dist)
        sum += x;
      j["fromCpu"] = {{"meanCm", r4(dist.empty() ? 0 : sum / double(dist.size()))},
                      {"p99Cm", r4(dist.empty() ? 0 : dist[size_t(0.99 * double(dist.size() - 1))])},
                      {"maxCm", r4(dist.empty() ? 0 : dist.back())}};
    }
    drapes.push_back(j);
  }
  return drapes;
}

// The colliders at fraction t of the transition, as the GPU solver puts them (kinematicAt): a skinned
// one along its bones from the memorized pose, another along straight lines.
void collidersAt(const cloth::Job& job, const cloth::Model& model, float t, std::vector<float>& out)
{
  for(size_t mi = 0; mi < job.surfaces.size(); mi++)
  {
    const cloth::Surface& m = job.surfaces[mi];
    if(m.kind != cloth::Surface::Kind::eCollider)
      continue;
    float* to = out.data() + size_t(model.base[mi]) * 3;
    if(m.skinned())
    {
      std::vector<fv::Mat4> bones(m.bones.size());
      for(size_t b = 0; b < bones.size(); b++)
        bones[b] = fv::physics::sweepAt(fv::physics::poseOf(fv::Mat4{}), fv::physics::poseOf(m.bones[b]), t);
      cloth::skin(m, bones, to);
    }
    else
      for(size_t k = 0; k < m.rest.size(); k++)
        to[k] = (1 - t) * m.rest[k] + t * m.posed[k];
  }
}

// --grid: the collision grid over the transition, the builder against its reference.
Json gridRun(const cloth::Job& job, const cloth::Model& model)
{
  std::vector<int32_t> corners;
  for(int t : model.staticTris)
  {
    const cloth::Face& f = model.tris[size_t(t)];
    corners.insert(corners.end(), {f.a, f.b, f.c});
  }
  const cloth::Island island(job.surfaces, job.options);
  const int           frames = std::max(1, int(std::lround(island.transitionSeconds() / job.options.frameDt)));
  std::vector<float>  start = model.rest, end = model.rest;
  fv::physics::SweepGrid fast, ref;
  double fastMs = 0, refMs = 0, entries = 0, used = 0, slots = 0;
  bool   same   = true;
  collidersAt(job, model, 0, end);
  for(int f = 0; f < frames; f++)
  {
    start = end;
    collidersAt(job, model, float(f + 1) / float(frames), end);
    auto t0 = std::chrono::steady_clock::now();
    fv::physics::buildSweepGrid(corners, start.data(), end.data(), 2.f, fast);
    fastMs += since(t0);
    t0 = std::chrono::steady_clock::now();
    fv::physics::buildSweepGridReference(corners, start.data(), end.data(), 2.f, ref);
    refMs += since(t0);
    same = same && fv::physics::sameSweepGrids(ref, fast);
    entries += double(fast.entries);
    used += double(fast.used);
    slots += double(fast.cells.size());
  }
  // what a lookup costs in the last frame's table, as the kernels probe it: slots read to find a cell, and
  // to find that a cell next to one isn't there
  double hitProbes = 0, missProbes = 0, hits = 0, misses = 0;
  const uint32_t mask = fast.mask();
  for(const fv::physics::SweepCell& c : fast.cells)
  {
    if(c.start < 0)
      continue;
    for(int dx = -1; dx <= 1; dx++)
      for(int dy = -1; dy <= 1; dy++)
        for(int dz = -1; dz <= 1; dz++)
        {
          const int x = c.x + dx, y = c.y + dy, z = c.z + dz;
          uint32_t  at = fv::physics::sweepCellHash(x, y, z) & mask;
          int       read = 0;
          bool      found = false;
          for(uint32_t probe = 0; probe <= mask; probe++)
          {
            const fv::physics::SweepCell& e = fast.cells[(at + probe) & mask];
            read++;
            if(e.start < 0)
              break;
            if(e.x == x && e.y == y && e.z == z)
            {
              found = true;
              break;
            }
          }
          (found ? hitProbes : missProbes) += read;
          (found ? hits : misses) += 1;
        }
  }
  return {{"frames", frames},       {"faces", corners.size() / 3},          {"entriesPerFrame", r4(entries / frames)},
          {"slotsReadToFind", r4(hitProbes / std::max(1.0, hits))}, {"slotsReadToMiss", r4(missProbes / std::max(1.0, misses))},
          {"cellsPerFrame", r4(used / frames)}, {"slotsPerFrame", r4(slots / frames)}, {"msPerFrame", r4(fastMs / frames)},
          {"referenceMsPerFrame", r4(refMs / frames)}, {"same", same}};
}

// The CPU island's settle, frame by frame as Island::settle runs it, so the frames can be counted.
std::vector<std::vector<float>> settleCpu(const cloth::Job& job, Json& report)
{
  const auto     t0 = std::chrono::steady_clock::now();
  cloth::Island  sim(job.surfaces, job.options);
  const double   buildMs    = since(t0);
  const auto&    o          = job.options;
  const int      transition = std::max(1, int(std::lround(sim.transitionSeconds() / o.frameDt)));
  const int      hold       = std::max(1, int(std::ceil(o.maxHold / o.frameDt)));
  int            frames     = 0;
  const auto     t1         = std::chrono::steady_clock::now();
  for(int f = 0; f < transition; f++, frames++)
    sim.frame(float(f) / float(transition), float(f + 1) / float(transition));
  for(int h = 0; h < hold; h++)
  {
    sim.frame(1, 1, 6.f);
    frames++;
    if(sim.lastMove() < o.restMove)
      break;
  }
  const double ms = since(t1);
  report          = {{"frames", frames}, {"substeps", sim.substepsPerFrame()}, {"buildMs", r4(buildMs)}, {"settleMs", r4(ms)}, {"msPerFrame", r4(ms / frames)}};
  return sim.positions();
}

}  // namespace


namespace {

Json selfReport(const fv::physics::cloth::Model& model, const float* pos = nullptr)
{
  if(!pos)
    pos = model.rest.data();
  namespace cloth = fv::physics::cloth;
  Json out = Json::array();
  // every face of a mesh whose own vertices meet it, at the rest shape, in cells of 2 cm
  std::vector<int32_t> corners;
  std::vector<int>     faceOf;
  for(int t : model.dynamicTris)
  {
    const cloth::Face& tri = model.tris[size_t(t)];
    if(!tri.self)
      continue;
    corners.insert(corners.end(), {tri.a, tri.b, tri.c});
    faceOf.push_back(t);
  }
  fv::physics::SweepGrid grid;
  const float            cell = 2.f;
  if(!corners.empty())
    fv::physics::buildSweepGrid(corners, pos, pos, cell, grid);
  const auto inRing2 = [&](int i, int v) {
    const int* first = model.ring2.data() + model.ring2Start[size_t(i)];
    const int* last  = model.ring2.data() + model.ring2Start[size_t(i) + 1];
    return std::binary_search(first, last, v);
  };
  const auto at = [&](int v) { return glm::vec3(pos[size_t(v) * 3], pos[size_t(v) * 3 + 1], pos[size_t(v) * 3 + 2]); };
  const int  meshes = model.base.empty() ? 0 : int(model.base.size());
  for(int mi = 0; mi < meshes; mi++)
  {
    const int first = model.base[size_t(mi)], last = mi + 1 < meshes ? model.base[size_t(mi) + 1] : model.nv;
    int       freeHere = 0, selfFaces = 0;
    double    edgeSum = 0;
    int       edges   = 0;
    float     offLo = 1e9f, offHi = 0;
    for(int t : model.dynamicTris)
      if(model.tris[size_t(t)].mesh == mi && model.tris[size_t(t)].self)
        selfFaces++;
    for(const cloth::Distance& e : model.edges)
      if(e.kind == cloth::DistanceKind::eStretch && model.meshOf[size_t(e.a)] == mi)
      {
        edgeSum += e.rest;
        edges++;
      }
    for(int v = first; v < last; v++)
    {
      if(model.dynamic[size_t(v)] && model.inv[size_t(v)] > 0)
        freeHere++;
      offLo = std::min(offLo, model.offset[size_t(v)]);
      offHi = std::max(offHi, model.offset[size_t(v)]);
    }
    if(freeHere == 0)
      continue;
    std::atomic<int>   withinGap{0}, withinTwice{0}, mostFaces{0};
    std::atomic<int>   deepestMilli{0};
    const uint32_t     mask = grid.cells.empty() ? 0 : grid.mask();
    fv::jobs::parallelFor(size_t(last - first), 512, [&](size_t begin, size_t end) {
      std::vector<int> seen;
      for(size_t k = begin; k < end; k++)
      {
        const int i = first + int(k);
        if(!model.dynamic[size_t(i)] || model.inv[size_t(i)] <= 0 || !model.collides[size_t(i)] || grid.cells.empty())
          continue;
        const glm::vec3 q     = at(i);
        const float     reach = 2 * model.offset[size_t(i)] + 0.5f;
        int             nGap = 0, nTwice = 0;
        float           deepest = 0;
        seen.clear();
        for(int ix = int(std::floor((q.x - reach) / cell)); ix <= int(std::floor((q.x + reach) / cell)); ix++)
          for(int iy = int(std::floor((q.y - reach) / cell)); iy <= int(std::floor((q.y + reach) / cell)); iy++)
            for(int iz = int(std::floor((q.z - reach) / cell)); iz <= int(std::floor((q.z + reach) / cell)); iz++)
            {
              const uint32_t s = fv::physics::sweepCellHash(ix, iy, iz) & mask;
              for(uint32_t probe = 0; probe <= mask; probe++)
              {
                const fv::physics::SweepCell& c = grid.cells[(s + probe) & mask];
                if(c.start < 0)
                  break;
                if(c.x != ix || c.y != iy || c.z != iz)
                  continue;
                for(int n = 0; n < c.count; n++)
                {
                  const int f = grid.items[size_t(c.start + n)];
                  if(std::find(seen.begin(), seen.end(), f) != seen.end())
                    continue;
                  seen.push_back(f);
                  const cloth::Face& tri = model.tris[size_t(faceOf[size_t(f)])];
                  if(tri.mesh != mi || tri.a == i || tri.b == i || tri.c == i || inRing2(i, tri.a) || inRing2(i, tri.b) || inRing2(i, tri.c))
                    continue;
                  const glm::vec3 a = at(tri.a), b = at(tri.b), cc = at(tri.c);
                  glm::vec3       nrm = glm::cross(b - a, cc - a);
                  const float     len = glm::length(nrm);
                  if(len < 1e-8f)
                    continue;
                  nrm /= len;
                  // the closest point on the face, as the kernels find it
                  const glm::vec3 ab = b - a, ac = cc - a, aq = q - a;
                  const float d00 = glm::dot(ab, ab), d01 = glm::dot(ab, ac), d11 = glm::dot(ac, ac), d20 = glm::dot(ab, aq), d21 = glm::dot(ac, aq);
                  const float den = d00 * d11 - d01 * d01;
                  float       bv = 0, cv = 0;
                  if(den > 1e-12f)
                  {
                    bv = std::clamp((d11 * d20 - d01 * d21) / den, 0.f, 1.f);
                    cv = std::clamp((d00 * d21 - d01 * d20) / den, 0.f, 1.f);
                    if(bv + cv > 1)
                    {
                      const float sum = bv + cv;
                      bv /= sum;
                      cv /= sum;
                    }
                  }
                  const glm::vec3 d    = q - (a + ab * bv + ac * cv);
                  const float     dist = glm::length(d), sd = glm::dot(d, nrm);
                  if(std::fabs(sd) < 0.6f * dist)
                    continue;  // past the edge
                  const float gap = std::max(tri.offset, model.offset[size_t(i)]);
                  if(dist < gap)
                  {
                    nGap++;
                    deepest = std::max(deepest, gap - dist);
                  }
                  if(dist < 2 * gap)
                    nTwice++;
                }
                break;
              }
            }
        if(nGap > 0)
          withinGap++;
        if(nTwice > 0)
          withinTwice++;
        int most = mostFaces.load();
        while(nGap > most && !mostFaces.compare_exchange_weak(most, nGap))
        {
        }
        int deep = deepestMilli.load();
        while(int(deepest * 1000) > deep && !deepestMilli.compare_exchange_weak(deep, int(deepest * 1000)))
        {
        }
      }
    });
    out.push_back({{"mesh", mi},
                   {"free", freeHere},
                   {"selfFaces", selfFaces},
                   {"meanEdgeCm", edges ? edgeSum / edges : 0.0},
                   {"offsetCm", {offLo, offHi}},
                   {"verticesWithinGapOfOwnFace", withinGap.load()},
                   {"verticesWithinTwiceGap", withinTwice.load()},
                   {"mostOwnFacesOnOneVertex", mostFaces.load()},
                   {"deepestCm", deepestMilli.load() / 1000.0}});
  }
  return out;
}

}  // namespace

int main(int argc, char** argv)
{
  if(argc < 2)
  {
    std::cerr << "fv_cloth_bench <job> [out.json] [--solvers cpu,gpu,cluster] [--cluster 1024] [--local 2048] [--analyse]\n";
    return 2;
  }
  std::string              jobFile = argv[1], outFile;
  std::vector<std::string> solvers = {"cpu", "gpu", "cluster"};
  std::vector<int>         sizes   = {1024};
  int                      local   = 2048;
  bool                     onlyAnalyse = false, onlyGrid = false, onlySelf = false, trace = false;
  for(int i = 2; i < argc; i++)
  {
    const std::string a = argv[i];
    if(a == "--solvers" && i + 1 < argc)
      solvers = split(argv[++i]);
    else if(a == "--cluster" && i + 1 < argc)
    {
      sizes.clear();
      for(const auto& s : split(argv[++i]))
        sizes.push_back(std::stoi(s));
    }
    else if(a == "--local" && i + 1 < argc)
      local = std::stoi(argv[++i]);
    else if(a == "--analyse")
      onlyAnalyse = true;
    else if(a == "--grid")
      onlyGrid = true;
    else if(a == "--self")
      onlySelf = true;
    else if(a == "--trace")
      trace = true;
    else if(outFile.empty())
      outFile = a;
  }
  const auto job = cloth::readJob(jobFile);
  if(!job)
  {
    std::cerr << "not a cloth job: " << jobFile << "\n";
    return 1;
  }
  const cloth::Model model = cloth::Model::build(job->surfaces, job->options.air, job->options.reachLinks);
  size_t             free  = model.freeVerts.size();
  Json               out   = {{"job", jobFile}, {"vertices", model.nv}, {"free", free}, {"rows", model.edges.size()}, {"faces", model.tris.size()}};
  Json               cuts  = Json::array();
  for(int size : sizes)
    cuts.push_back(analyse(model, size, local));
  out["clusters"] = cuts;
  std::cout << out.dump(1) << std::endl;
  const auto write = [&] {
    if(!outFile.empty())
      std::ofstream(outFile) << out.dump(1);
  };
  if(onlyAnalyse)
  {
    write();
    return 0;
  }
  if(onlySelf)
  {
    out["self"] = selfReport(model);
    std::cout << out["self"].dump(1) << std::endl;
    write();
    return 0;
  }
  if(onlyGrid)
  {
    out["grid"] = gridRun(*job, model);
    std::cout << "grid " << out["grid"].dump() << std::endl;
    write();
    return 0;
  }

  Json                            runs = Json::object();
  std::vector<std::vector<float>> cpu;
  if(std::find(solvers.begin(), solvers.end(), "cpu") != solvers.end())
  {
    Json report;
    cpu            = settleCpu(*job, report);
    report["drapes"] = judge(*job, model, cpu, nullptr);
    runs["cpu"]    = report;
    std::cout << "cpu " << report.dump() << std::endl;
  }

  const bool anyGpu = std::any_of(solvers.begin(), solvers.end(), [](const std::string& s) { return s != "cpu"; });
  if(anyGpu)
  {
    nvvk::ContextInitInfo info;
    info.queues = {VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT};
    nvvk::Context context;
    if(context.init(info) != VK_SUCCESS)
    {
      std::cerr << "no Vulkan device\n";
      return 1;
    }
    nvapp::ApplicationCreateInfo appInfo;
    appInfo.name            = "fv_cloth_bench";
    appInfo.headless        = true;
    appInfo.persistSettings = false;
    appInfo.useMenu         = false;
    appInfo.instance        = context.getInstance();
    appInfo.device          = context.getDevice();
    appInfo.physicalDevice  = context.getPhysicalDevice();
    appInfo.queues          = context.getQueueInfos();
    nvapp::Application app;
    app.init(appInfo);
    nvvk::ResourceAllocator alloc;
    alloc.init({
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = context.getPhysicalDevice(),
        .device           = context.getDevice(),
        .instance         = context.getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(context.getPhysicalDevice(), &props);
    out["device"] = props.deviceName;
    for(const std::string& name : solvers)
    {
      if(name == "cpu")
        continue;
      const bool clustered = name == "cluster" || name == "coloured";
      for(int size : clustered ? sizes : std::vector<int>{0})
      {
        ClothGpu::Options gpu;
        gpu.clusterSize  = size;
        gpu.clusterLocal = local;
        gpu.crossing     = name == "cluster" ? ClothGpu::Options::Crossing::eHalo : ClothGpu::Options::Crossing::eColoured;
        gpu.timestamps   = true;
        const auto t0    = std::chrono::steady_clock::now();
        {
          ClothGpu sim(&alloc, &app, job->surfaces, job->options, gpu);
          if(trace)
          {
            // a frame at a time, and after each the cloth faces under the cloth's vertices
            int frameNo = 0;
            while(!sim.settleSome(1e-9) && !sim.failed())
            {
              if(sim.stats().frames == 0)
                continue;  // still preparing
              const auto         ps = sim.positions();
              std::vector<float> all;
              for(const auto& v : ps)
                all.insert(all.end(), v.begin(), v.end());
              const Json r = selfReport(model, all.data());
              std::cout << "trace frame " << ++frameNo << " move " << sim.lastMove();
              for(const Json& m : r)
                std::cout << " | mesh " << m["mesh"] << " gap " << m["verticesWithinGapOfOwnFace"] << " twice " << m["verticesWithinTwiceGap"] << " most "
                          << m["mostOwnFacesOnOneVertex"] << " deep " << m["deepestCm"];
              std::cout << std::endl;
            }
          }
          else
            sim.settle();
          const double ms = since(t0);
          const auto   s  = sim.stats();
          if(sim.failed())
          {
            runs[name + (size ? std::to_string(size) : "")] = {{"failed", true}};
            continue;
          }
          const auto pos    = sim.positions();
          Json       report = {{"frames", s.frames},
                               {"substeps", s.substeps},
                               {"settleMs", r4(ms)},
                               {"gpuMsPerFrame", r4(s.gpuMs / std::max(1, s.frames))},
                               {"collideMsPerFrame", r4(s.collideMs / std::max(1, s.frames))},
                               {"recoverMsPerFrame", r4(s.recoverMs / std::max(1, s.frames))},
                               {"holdMsPerFrame", r4(s.holdMs / std::max(1, s.frames))},
                               {"waitMsPerFrame", r4(s.waitMs / std::max(1, s.frames))},
                               {"skinMsPerFrame", r4(s.skinMs / std::max(1, s.frames))},
                               {"gridMsPerFrame", r4(s.gridMs / std::max(1, s.frames))},
                               {"gridCopyMsPerFrame", r4(s.gridCopyMs / std::max(1, s.frames))},
                               {"gridEntriesPerFrame", r4(double(s.gridEntries) / std::max(1, s.frames))},
                               {"gridSlotsPerFrame", r4(double(s.gridSlots) / std::max(1, s.frames))},
                               {"clothFaces", s.clothFaces},
                               {"clothGridMsPerFrame", r4(s.clothGridMs / std::max(1, s.frames))},
                               {"dispatchesPerFrame", r4(double(s.dispatches) / std::max(1, s.frames))},
                               {"dispatchesPerSubstep", r4(double(s.dispatches) / std::max(1, s.frames * std::max(1, s.substeps)))},
                               {"rows", s.rows}};
          if(size)
            report["cut"] = {{"clusters", s.clusters},         {"clusterColours", s.clusterColours}, {"crossingRows", s.crossingRows},
                             {"crossingColours", s.crossingColours}, {"haloRows", s.haloRows},           {"mostLocal", s.mostLocal}};
          report["drapes"] = judge(*job, model, pos, cpu.empty() ? nullptr : &cpu);
          // the same job again on the same simulation, which keeps what it made and starts over
          const auto t1   = std::chrono::steady_clock::now();
          const bool kept = sim.restart(job->surfaces, job->options);
          sim.settle();
          const double again = since(t1);
          const auto   pos2  = sim.positions();
          report["restart"]  = {{"kept", kept}, {"settleMs", r4(again)}, {"frames", sim.stats().frames}, {"sameDrape", !sim.failed() && pos2 == pos}};
          // the body dented: every seventh vertex of its rest cage 2 mm in along z, its target skinned again
          cloth::Job dented = *job;
          for(cloth::Surface& m : dented.surfaces)
          {
            if(m.kind != cloth::Surface::Kind::eCollider || !m.skinned())
              continue;
            for(size_t v = 0; v < m.vertexCount(); v += 7)
            {
              m.restLocal[v * 3 + 2] -= 0.2f;
              const fv::Vec3 p = m.toSpace.transformPoint({m.restLocal[v * 3], m.restLocal[v * 3 + 1], m.restLocal[v * 3 + 2]});
              m.rest[v * 3]     = float(p.x);
              m.rest[v * 3 + 1] = float(p.y);
              m.rest[v * 3 + 2] = float(p.z);
            }
            cloth::skin(m, m.bones, m.posed.data());
            break;
          }
          const auto t2 = std::chrono::steady_clock::now();
          sim.restart(dented.surfaces, dented.options);
          sim.settle();
          const double dentedMs = since(t2);
          const auto   pos3     = sim.positions();
          ClothGpu    fresh(&alloc, &app, dented.surfaces, dented.options, gpu);
          fresh.settle();
          report["dented"] = {{"settleMs", r4(dentedMs)}, {"sameAsNew", !sim.failed() && !fresh.failed() && pos3 == fresh.positions()},
                              {"movedFromUndented", pos3 != pos}};
          const std::string key = clustered ? name + std::to_string(size) : name;
          runs[key]             = report;
          std::cout << key << " " << report.dump() << std::endl;
        }
        vkDeviceWaitIdle(context.getDevice());
      }
    }
    alloc.deinit();
    app.deinit();
    context.deinit();
  }
  out["runs"] = runs;
  write();
  return 0;
}
