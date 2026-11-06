// The GPU cloth bench (docs/physics-plan.md step 13a): a cloth job, as fv_parity's dforce step writes
// it ("job"), settled on the CPU island, on the GPU solver a dispatch a colour (today's) and on its
// cluster solve, each timed, its drapes judged by the numbers fv_parity reports for them and set
// against the CPU's vertex by vertex. Needs a Vulkan device and nothing of the app.
//
//   fv_cloth_bench <job> [out.json] [--solvers cpu,gpu,cluster,coloured] [--cluster 1024[,512..]] [--local 2048]
//                  [--analyse] [--grid]
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

// the allocator's implementation, as renderer.cpp compiles it for the renderer
#define VMA_IMPLEMENTATION
#include <nvvk/resource_allocator.hpp>

#include <algorithm>
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
#include <nvvk/context.hpp>
#include <nvvk/resource_allocator.hpp>
#include <volk.h>

#include "fv_dforce_gpu.hpp"

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
  bool                     onlyAnalyse = false, onlyGrid = false;
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
        DforceGpu::Options gpu;
        gpu.clusterSize  = size;
        gpu.clusterLocal = local;
        gpu.crossing     = name == "cluster" ? DforceGpu::Options::Crossing::eHalo : DforceGpu::Options::Crossing::eColoured;
        gpu.timestamps   = true;
        const auto t0    = std::chrono::steady_clock::now();
        {
          DforceGpu sim(&alloc, &app, job->surfaces, job->options, gpu);
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
          DforceGpu    fresh(&alloc, &app, dented.surfaces, dented.options, gpu);
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
