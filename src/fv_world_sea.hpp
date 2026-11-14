#pragma once

//-------------------------------------------------------------------------------------------------------
// What grows on a planet's sea floor (fv_world.hpp). The backdrop's kit is a northern meadow's and a
// fir wood's and has nothing for the sea, so these are made here: small meshes from a seed, a few
// variants a role, placed as the land's plants are (fv_world_plants.hpp `seaStanding`).
//
//   seagrass  a tuft of ribbon leaves, in meadows on the shallows' sand
//   kelp      a stalk with blades up it, in groves where the water is cool and some metres deep
//   coral     branching, in clusters in warm shallows
//   sponge    a lumpy vase, or a cluster of tubes, deeper
//   searock   a lumpy boulder with flat faces, anywhere
//
// The generators are after those of Noniv's abyssal-garden (https://github.com/Noniv/abyssal-garden,
// MIT; source/src/world.js: meshLathe, meshCoral, meshTubes, meshChunk): a profile turned about an
// axis with lumps, branches as tapering tubes that fork, a cluster of leaning tubes with dark
// throats, a lumpy ball cut by planes. Written again here for this mesh format; the kelp and the
// seagrass are this file's own.
//
// A mesh stands on y = 0 with +y up, a metre or so high (`height`), and is drawn scaled. Its vertex
// colours are its own (a coral's pink, a tube's dark throat): the material's colour is white.
//
// Vulkan-free.
//-------------------------------------------------------------------------------------------------------

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

namespace fv::world {

// The sea floor's roles, in the order `seaStanding` counts them
inline constexpr std::array<std::string_view, 5> kSeaRoles = {"seagrass", "kelp", "coral", "sponge", "searock"};

struct SeaMesh
{
  uint8_t                role = 0;  // of kSeaRoles
  std::vector<glm::vec3> positions, normals;
  std::vector<glm::vec4> colors;
  std::vector<uint32_t>  indices;
  float                  height = 1.0F;  // metres, as made
  bool                   leaves = false; // thin leaves seen from both sides
};

// A role's variants, in kSeaRoles' order: the same for a seed every time
std::vector<SeaMesh> seaMeshes(int seed);

}  // namespace fv::world
