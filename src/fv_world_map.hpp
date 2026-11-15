#pragma once

//-------------------------------------------------------------------------------------------------------
// A planet's base map (fv_world.hpp): what a point of the planet cannot say by itself, worked out once
// over the whole of it on a cube map and read back by the stages.
//
// The bake, in order (`bake`):
//
// - The land as the stages before it have it, a height a cell (as finely as a cell holds it:
//   kMapLandFootprint).
// - Rain. The wind blows in belts by latitude; it takes up water over the sea and lets it fall over
//   the land, more where the land rises under it, so a range is wet on its windward side and dry
//   behind. On a coarser grid of its own: the weather is large.
// - Basins. Every hollow of the land is filled to where it spills (a priority flood from the coast),
//   so all of it drains. A deep one keeps its shape and holds a lake, whose water stands some way
//   under the brim: the map's cells are tens of metres, and the ground's finer relief, which the map
//   does not see, dips under a lake that is brim-full all round it. A shallow one is ground.
// - Wear. The rain that falls on a cell runs to its lowest neighbour, and what runs through a cell
//   cuts it down by how much water that is and how steep its way (the stream power law, solved
//   implicitly from the sea up: Braun and Willett 2013), against a land that rises where it stood
//   high. No slope stays steeper than loose ground holds. A few tens of rounds leave valleys that
//   branch up into the heights.
// - Floors. A valley the wear cut is a V; what its river carries fills it flat across, deeper the more
//   water runs in it, up to where its sides stand over the fill. Not simulated: a depth by its water.
// - Rivers. Where enough water runs, a river: a smooth curve from cell to cell, through each cell where
//   the ground as it is drawn lies lowest across its valley, with a width by its water, kept as curves
//   so its bed is cut and its surface drawn as finely as the ground is, whatever the map's cells are.
//   Its level follows that ground down (the stages' land finer than the cells, the wear's, the
//   gullies'), a little under it and never rising, so it cuts through what stands in its way, a step
//   of a cliff or a ridge, and never runs on a bank of its own; its banks are a valley's sides, eased
//   into the land. On a gentle floor with room it meanders: its water swings from side to side of its
//   curve in waves some eleven of its widths long, deeper and wider at a bend's outside, with a low
//   bar inside. Near the sea's level a river comes down to it early, and where the ground beside it is
//   under the sea it is the sea (a shore's pools are finer than the map's cells, and a river a little
//   over them would stand out of them). Across the dry ground of a lake's hollow a river runs at the
//   lake's level, cutting its way in or out.
//
// What is kept (BaseMap): what the wear and the floors did to the land (and that again at half the
// cells, and half again, for what draws coarsely), the rain, the lakes' levels, the worn land's broad
// slope (before the rivers, which follow the gullies laid by it), and the rivers' curves by cell.
//
// The cube's faces are grids of `cells` squares along an edge, laid as fv::world::direction lays a
// face (even in angle). A field is kept with a gutter round each face, filled from the faces beyond,
// so a lookup near an edge reads straight on. At the eight corners three faces meet and no gutter is
// right: the map is level there, and rivers near one are found from a list.
//
// Vulkan-free; the bake uses the job pool.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "fv_world.hpp"

namespace fv::world {

inline constexpr int   kMapGutter = 4;
inline constexpr float kDryCell   = -1.0e30F;  // a cell with no lake
// The stages' land is sampled for the map at this share of a cell's footprint (at a cell's own, the
// noises leave out what is under four of it, which the ground is drawn with)
inline constexpr double kMapLandFootprint = 0.25;

// Where a direction is on a grid: its face, and its place in cells (a cell's middle at whole numbers)
struct MapPoint
{
  uint8_t face = 0;
  double  x = 0.0, y = 0.0;
};

// A cube's six faces as grids
class CubeGrid
{
public:
  static constexpr uint32_t kNone = 0xFFFFFFFFU;

  explicit CubeGrid(uint32_t cells);
  uint32_t cells() const { return m_n; }
  uint32_t count() const { return 6U * m_n * m_n; }
  uint32_t index(uint8_t face, uint32_t i, uint32_t j) const { return (uint32_t(face) * m_n + j) * m_n + i; }
  // The cell (i, j) of `face`, where one of the two may be off the face's edge: the cell of the face
  // beyond, as if the grid ran straight on. False where both are off (a corner): there is none.
  bool cell(uint8_t face, int i, int j, uint32_t& index) const;
  // A cell's eight neighbours, counter-clockwise from +i; kNone for a corner's missing one
  void around(uint32_t index, uint32_t out[8]) const;
  glm::dvec3 centre(uint32_t index) const;
  MapPoint   at(const glm::dvec3& direction) const;

  // A field with its gutters: each face (cells + 2 kMapGutter) squared
  uint32_t stride() const { return m_n + 2U * uint32_t(kMapGutter); }
  size_t   padded(uint8_t face, int i, int j) const
  {
    return (size_t(face) * stride() + size_t(j + kMapGutter)) * stride() + size_t(i + kMapGutter);
  }
  // `cells` (one a cell, by index) laid out so; a corner's gutter takes the face's nearest cell, or `corner`
  template <class T>
  void pad(const std::vector<T>& cells, std::vector<T>& out, bool nearest = true, T corner = T{}) const
  {
    const int n = int(m_n);
    out.resize(size_t(6) * stride() * stride());
    for(uint8_t face = 0; face < 6; face++)
      for(int j = -kMapGutter; j < n + kMapGutter; j++)
        for(int i = -kMapGutter; i < n + kMapGutter; i++)
        {
          uint32_t c = 0;
          if(cell(face, i, j, c))
            out[padded(face, i, j)] = cells[c];
          else
            out[padded(face, i, j)] = nearest ? cells[index(face, uint32_t(std::clamp(i, 0, n - 1)), uint32_t(std::clamp(j, 0, n - 1)))] : corner;
        }
  }

private:
  struct Edge
  {
    uint8_t  face    = 0;      // the face beyond
    bool     acrossU = false;  // its u runs away from the shared edge (else its v)
    bool     high    = false;  // the edge is at its far end of that
    uint32_t along   = 0;      // the cell along the edge
  };
  uint32_t          m_n = 0;
  std::vector<Edge> m_edges;  // [face][side][along]
};

// A stretch of a river: a curve from `a` to `c` bent toward `b` (directions from the planet's centre).
// The water's level at its ends and its quarter points, straight between (it follows the ground down,
// which a curve would not); half its width at the ends and the middle, bent as the curve is.
// Its water does not run down that curve but swings from side to side of it, as a river on a valley's
// floor meanders: `wander` metres either way at its ends and middle (bent so), in waves whose phase is
// `phase` at its ends (radians, rising upstream, the same at a joint), some kMeanderWaves widths long.
inline constexpr int kRiverLevels = 5;
struct RiverPiece
{
  glm::dvec3 a{0.0}, b{0.0}, c{0.0};
  float      level[kRiverLevels] = {};
  float      half[3]             = {0.0F, 0.0F, 0.0F};
  float      wander[3]           = {0.0F, 0.0F, 0.0F};
  float      phase[2]            = {0.0F, 0.0F};
};
// Where its water's middle is a share `t` of the way along it, for a planet of `radius`
glm::dvec3 riverMiddleAt(const RiverPiece& piece, double t, double radius);
// The water's level a share `t` of the way along a piece
inline double riverLevelAt(const RiverPiece& p, double t)
{
  const double x = std::clamp(t, 0.0, 1.0) * double(kRiverLevels - 1);
  const int    k = std::min(int(x), kRiverLevels - 2);
  return double(p.level[k]) + (double(p.level[k + 1]) - double(p.level[k])) * (x - double(k));
}

struct BaseMap
{
  explicit BaseMap(uint32_t cells, uint32_t rainCells) : grid(cells), rainGrid(rainCells) {}

  CubeGrid grid;
  double   cell = 0.0;  // metres across a cell at a face's middle (the largest)

  // What the wear added to the land, metres: levels[0] a cell of the grid, each next half as many
  struct Level
  {
    CubeGrid           grid;
    std::vector<float> added;  // with gutters
  };
  std::vector<Level> levels;

  CubeGrid           rainGrid;
  std::vector<float> rain;  // 0 dry to 1 wet, with gutters

  std::vector<float>      lake;    // a lake cell's water level, kDryCell elsewhere; with gutters
  // Which way the land falls and how steeply, seen broadly: its rise over its run as a vector along
  // the ground (the planet's axes), at the corners of a grid of squares (each face's own, a shared
  // edge's kept alike on both). No gutters: read between four corners, it is the same from either side
  // of an edge, which what is laid along it (the `gullies` stage) needs. slopes[0] has slopeCellsFor
  // squares along a face, each next half as many and a slope as much broader: a gully follows the
  // land's fall at its own size.
  struct Slope
  {
    uint32_t               cells = 0;
    std::vector<glm::vec3> at;  // (cells + 1) squared a face
  };
  std::vector<Slope> slopes;

  std::vector<int32_t>    river;   // a river cell's piece, -1 elsewhere; with gutters
  std::vector<uint8_t>    reach;   // by cell: a river runs within the lookup's window of it
  std::vector<RiverPiece> pieces;
  std::array<std::vector<uint32_t>, 8> corner;  // the pieces near each of the cube's corners

  // By cell, as the bake left them (the tests', and what later steps place things by)
  std::vector<float>    height;  // the land's, metres over the sea; a lake's bed
  std::vector<float>    flow;    // the rain that runs through, square metres of wet ground upstream
  std::vector<uint32_t> down;    // the cell it runs to; CubeGrid::kNone where it ends
  std::vector<uint8_t>  kind;    // MapCell
  uint32_t              lakes = 0;
};

enum MapCell : uint8_t
{
  eMapSea = 0,
  eMapLand,
  eMapLake,   // under a lake's water
  eMapBasin,  // the rest of a lake's hollow, over its water: ground that is not worn
};

// The cells along a face for a planet of `radius`: about 60 m a cell, between 64 and 1024
uint32_t mapCells(double radius);

// Bakes the map of `world`, whose stages are those that come before it. Null when stopped. `floors`:
// the valleys' floors filled (else they are as the wear cut them).
std::shared_ptr<const BaseMap> bake(const World& world, uint32_t cells, const std::atomic<bool>* stop = nullptr, bool floors = true);

// What the wear added at a direction, for something `footprint` metres across
double mapAdded(const BaseMap& map, const glm::dvec3& direction, double footprint);
// How wet the weather is there, 0 to 1
float mapRain(const BaseMap& map, const glm::dvec3& direction);
// The land's broad slope at a direction (BaseMap::slopes): uphill, as long as it is steep. `on`: where
// the direction is on the cube (locate). `level`: which of the slopes, the nearest there is.
glm::dvec3 mapSlope(const BaseMap& map, const FacePoint& on, const glm::dvec3& direction, int level = 0);
inline glm::dvec3 mapSlope(const BaseMap& map, const glm::dvec3& direction, int level = 0)
{
  return mapSlope(map, locate(direction), direction, level);
}
// Which of the slopes has squares nearest `metres` across
int slopeLevelFor(const BaseMap& map, double radius, double metres);
// The squares along a face of the finest of them, for a map of `cells`: some 600 m across on the
// largest planet
uint32_t slopeCellsFor(uint32_t cells);
// The inland water near
Inland mapInland(const BaseMap& map, double radius, const glm::dvec3& direction);
// What the `gullies` stage cuts into the land at a direction, for something `footprint` across, the
// land being `before` (its stages' and the wear's), by `map`'s slopes: in metres, up or down
// (fv_world.cpp). The bake asks it of the ground its rivers run on.
double gulliesAt(const World& world, const BaseMap& map, const glm::dvec3& direction, double footprint, const Sample& before);

// The water's surface for what `inland` says, as its mesh draws it; kDry with none near
double inlandSheet(const Inland& inland, double footprint);
// The ground there, from what it would be with no water (`natural`): a river's bed and banks cut
// into it, kept over a lake's level outside the lake
double inlandGround(const Inland& inland, double natural, double footprint);
// 0..1: how much of a river too narrow for `footprint` shows as colour
float inlandPainted(const Inland& inland, double footprint);
// 0..1: how much of the land's own fine relief (its bumps) stands there: none in the water, all of it
// a little way past a river's bank and out of a lake's cells. It eases in, so the land has no step
// where a river's reach ends. With `rivers` false, a lake's alone (the gullies, which run into a river).
double inlandRelief(const Inland& inland, bool rivers = true);
// 0..1: how near the water of a river or a lake: 1 at its edge, easing to 0 by the end of a river's
// reach and out of a lake's cells (nothing that reads it steps there)
double inlandNear(const Inland& inland);

}  // namespace fv::world
