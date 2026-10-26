#include <figure/materials/fabric.hpp>

#include <algorithm>
#include <array>

namespace fv::mat {

namespace {

// Thread counts and relief of everyday cloth. A tile is the threads the app's generator draws on it
// (src-tauri/src/fabrics.rs), so its size is those threads at the cloth's count; the relief is the
// yarns' rise over the tile.
constexpr std::array kFabrics = {
    // 3/1 twill, 64 ends at about 30 per cm (48 picks at 23), yarns rising about 0.2 mm
    Fabric{"denim", 0.021f, 0.0095f, 2.0f},
    // heavy plain weave, 32 threads at about 14 per cm, coarse yarn rising about 0.3 mm
    Fabric{"canvas", 0.023f, 0.013f, 2.2f},
    // plain weave, 48 threads at about 20 per cm with slubs, rising about 0.15 mm
    Fabric{"linen", 0.024f, 0.0063f, 2.0f},
    // single jersey, 32 wales 0.9 mm apart (40 courses), loops rising about 0.3 mm
    Fabric{"knit", 0.029f, 0.010f, 2.2f},
    // 8-shaft satin, 96 ends at about 80 per cm, barely raised; smoother than the surface alone
    Fabric{"satin", 0.012f, 0.0017f, 1.4f},
    // 16 wales of 11 per inch (2.3 mm), ribs about 0.6 mm high
    Fabric{"corduroy", 0.037f, 0.016f, 2.0f},
    // pebbled grain, about 14 cells of 1.4 mm, about 0.08 mm deep
    Fabric{"leather", 0.020f, 0.004f, 1.8f},
};

// The tiling with no measured surface size: about a garment panel's worth of weave per UV unit.
constexpr float kUnknownSizeTiles = 64;

}  // namespace

std::span<const Fabric> fabrics()
{
  return kFabrics;
}

const Fabric* findFabric(std::string_view key)
{
  for(const Fabric& f : kFabrics)
    if(key == f.key)
      return &f;
  return nullptr;
}

std::string fabricHeightMap(const Fabric& f)
{
  return std::string(kFabricDir) + f.key + "-height.png";
}

std::string fabricRoughnessMap(const Fabric& f)
{
  return std::string(kFabricDir) + f.key + "-rough.png";
}

std::optional<DetailLayer> fabricDetail(const Edits& e, float metresPerUv)
{
  const Fabric* f = e.fabric ? findFabric(*e.fabric) : nullptr;
  if(!f)
    return std::nullopt;
  const float scale = std::max(e.fabricScale.value_or(1.f), 0.01f);
  const float tiles = metresPerUv > 0 ? metresPerUv / (f->tileMetres * scale) : kUnknownSizeTiles / scale;
  DetailLayer d;
  d.map          = fabricHeightMap(*f);
  d.heightMap    = true;
  d.bumpHeight   = f->relief;
  d.strength     = e.fabricStrength.value_or(1.f);
  d.weight       = 1;
  d.tiles        = {std::clamp(tiles, 1.f, 4096.f), std::clamp(tiles, 1.f, 4096.f)};
  d.roughness    = f->roughness;
  d.roughnessMap = fabricRoughnessMap(*f);
  return d;
}

}  // namespace fv::mat
