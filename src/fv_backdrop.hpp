#pragma once

//-------------------------------------------------------------------------------------------------------
// The backdrop (the lighting's `backdrop` section, protocol 64; the app's Light tab › Backdrop): ground
// that rolls away from where the figures stand, planted with a kit's plants. A meadow: grass and
// flowers underfoot, saplings further out, and trees in groves that close the horizon.
//
// Everything here follows from the spec alone, so a scene file keeps only the spec: the ground's height
// (`height`), its mesh (`terrain`), and where each plant stands (`scatter`), by a seed. The host
// (fv_host_backdrop.cpp) draws each plant variant once with its places as instancing copies
// (rs::MeshInstance::copies), so a plant's geometry is stored once however many stand, and nothing is an
// entity or a node of its own.
//
// The plants are a kit's, in a folder the app fills (Poly Haven's CC0 models, their cutout maps put
// back): `kit.json` names each file's role (KitPlant). Which roles a kind plants, how thickly and
// where, is `layers`.
//
// Vulkan-free; distances in metres, the figures' floor at y = 0.
//-------------------------------------------------------------------------------------------------------

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace fv::backdrop {

struct Spec
{
  std::string kind    = "none";  // none | meadow
  int         seed    = 1;       // which meadow: the hills and where everything stands
  double      size    = 60.0;    // metres across the ground planted underfoot (grass, flowers, saplings)
  double      density = 1.0;     // of every layer's own, 0.25 to 2
  double      relief  = 1.0;     // how hilly the ground is beyond the clearing, 0 flat to 2
  bool        operator==(const Spec&) const = default;
};

inline bool planted(const Spec& s)
{
  return s.kind != "none";
}

// The ground stays flat (the floor's height) within this of the origin, where the figures stand, and
// rises into its hills over kClearingEdge more
inline constexpr float kClearing     = 5.0F;
inline constexpr float kClearingEdge = 25.0F;
// Where the ground ends: far enough for its hills to be the horizon's
inline constexpr float kHorizon = 900.0F;

// How far the tufts beyond the planted grass reach from where one looks: a standing eye sees the
// ground between them closed by then, and the trees' groves have closed too
inline constexpr float kTufts = 140.0F;

// The ground's height over the floor at (x, z)
float height(const Spec& spec, float x, float z);

// The ground as a mesh: a disc, its rings fine underfoot and coarser with distance out to kHorizon, with
// smooth normals and UVs that tile every `tile` metres
struct Terrain
{
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec2> uvs;
  // 0..1 a vertex, in drifts some 40 m across: where the ground is drier. A tiled texture is one
  // colour from far off; the kit's two tints mixed by this give the distance its patches
  std::vector<float>    shades;
  std::vector<uint32_t> indices;
};
Terrain terrain(const Spec& spec, float tile);

// A layer of one role's plants
struct Layer
{
  std::string role;                  // the kit's role it plants: grass, flower, sapling, tree
  double      perSquareMetre = 1.0;  // where it stands thickest, before the spec's density
  float       from           = 0.0F; // none nearer the origin than this
  float       to             = 30.0F; // nor further
  // Past `thin` metres the layer thins as 1 / (1 + (d / thin)^2) and each plant grows by the root of
  // that, up to `grow` times, so distant ground keeps its cover with far fewer plants. 0: even.
  float thin = 0.0F;
  float grow = 1.0F;
  // Thinning as 1 / (1 + d / thin) instead: the rate at which plants seen from standing height go on
  // hiding the ground between them, since each hides a strip of it that lengthens with distance
  bool thinsGently = false;
  float fade = 0.0F;  // metres before `to` over which it thins out to nothing, so it has no edge
  float fadeIn = 0.0F;  // metres after `from` over which it comes in
  // Round the nearest focus (scatter's `foci`: where the scene is, where the camera stands) and not
  // round the origin: `from`, `to` and the thinning are by the distance to it. Grass is wherever one
  // looks from; the trees stay where they are
  bool follows = false;
  // In drifts and groves: the share of the ground the layer stands on at all (1 everywhere), by a noise
  // `patch` metres across; and how much nearer its far edge it fills in whatever the noise (0 none),
  // so trees close the horizon
  float cover    = 1.0F;
  float patch    = 20.0F;
  float closeFar = 0.0F;
  float scaleMin = 0.8F;
  float scaleMax = 1.25F;
  float sink     = 0.0F;  // metres into the ground, times its scale: a trunk's foot on a slope
};
std::vector<Layer> layers(const Spec& spec);

// A focus for a point (x, z): the nearest on a grid of focusStep metres, so a camera's small moves
// replant nothing, and the same view is the same meadow whichever way it was come to
float     focusStep(const Spec& spec);
glm::vec2 focusNear(const Spec& spec, glm::vec2 at);

// Where a layer's plants stand: per variant of the role (the kit's meshes for it, `variants` of them),
// each plant's world matrix: on the ground, turned about the vertical and scaled. `layer` is its index
// in layers(), so two layers of one spec don't share their places. `foci`, for a layer that follows:
// (x, z) each, none meaning the origin. A plant's place is the world's, whatever the foci: another
// focus shows other plants of the same meadow, and moves none.
std::vector<std::vector<glm::mat4>> scatter(const Spec&                spec,
                                            const Layer&               layer,
                                            uint32_t                   layerIndex,
                                            uint32_t                   variants,
                                            std::span<const glm::vec2> foci = {});

// How many plants scatter() places, about: for a notice before a heavy one is made
uint64_t expectedCount(const Spec& spec, const Layer& layer);

}  // namespace fv::backdrop
