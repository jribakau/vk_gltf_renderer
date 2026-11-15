#pragma once

//-------------------------------------------------------------------------------------------------------
// A map of a planet (fv_world.hpp), as an author reads one: a window of latitude and longitude drawn
// north up, each pixel the planet's ground at its middle in the ground's own colours, shaded by its
// slope as if lit from the north-west, the sea darker the deeper it is, its lakes, and its rivers and
// roads drawn over it as lines a pixel wide at the least (most are narrower than one). What stands
// on it and what its file says (places, regions, features) is for whoever shows the image to draw
// over it: chartPoint says where a direction is in it.
//
// Latitude and longitude as fv_world_features.hpp's directionOf has them: degrees, +y north,
// longitude 0 on +x and 90 on +z. A window may reach past ±180 of longitude, and is taken round.
// Pixels are as wide as the window's longitude over its width, so a window is true to shape at its
// middle when its longitude spans its latitude's times its width over its height, over the cosine of
// its middle's latitude (the app's to choose).
//
// Vulkan-free; chartImage uses the job pool.
//-------------------------------------------------------------------------------------------------------

#include <atomic>
#include <cstdint>
#include <vector>

#include "fv_world.hpp"

namespace fv::world {

struct ChartView
{
  double west = -180.0, east = 180.0, south = -90.0, north = 90.0;  // degrees
  int    width = 1024, height = 512;                                  // pixels
  bool   rivers = true, roads = true;                                 // drawn over the ground
};

// A direction's latitude and longitude in degrees, directionOf's inverse (the longitude -180 to 180)
glm::dvec2 latitudeLongitudeOf(const glm::dvec3& direction);
// The direction at a point of the image: `x` pixels from its west edge, `y` from its north
glm::dvec3 chartDirection(const ChartView& view, double x, double y);
// Where a direction is in the image, pixels as chartDirection's; its longitude taken round to the
// nearest the window's middle. Off the image's edges when it is outside the window.
glm::dvec2 chartPoint(const ChartView& view, const glm::dvec3& direction);
// Metres a pixel spans north to south
double chartPixel(const World& world, const ChartView& view);

// The image: width by height pixels of RGB, its rows from the north. Empty when stopped.
std::vector<uint8_t> chartImage(const World& world, const ChartView& view, const std::atomic<bool>* stop = nullptr);

}  // namespace fv::world
