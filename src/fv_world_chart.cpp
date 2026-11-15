// A map of a planet: see fv_world_chart.hpp.

#include "fv_world_chart.hpp"

#include <algorithm>
#include <cmath>

#include <figure/jobs/pool.hpp>

#include "fv_world_map.hpp"
#include "fv_world_places.hpp"

namespace fv::world {

namespace {

constexpr double kPi     = 3.14159265358979323846;
constexpr double kDegree = kPi / 180.0;

// The light the slopes are shaded by: from the north-west and high (east, north, up)
const glm::dvec3 kLight = glm::normalize(glm::dvec3(-1.0, 1.0, 1.1));

// Linear colours
const glm::dvec3 kShallow{0.030, 0.105, 0.170};
const glm::dvec3 kDeep{0.008, 0.028, 0.075};
const glm::dvec3 kInland{0.025, 0.110, 0.200};
const glm::dvec3 kRiver{0.030, 0.140, 0.260};
constexpr double kDeepAt = 2500.0;  // metres down where the sea is all kDeep

double middleLongitude(const ChartView& v)
{
  return 0.5 * (v.west + v.east);
}

}  // namespace

glm::dvec2 latitudeLongitudeOf(const glm::dvec3& d)
{
  const glm::dvec3 u = glm::normalize(d);
  return {std::asin(std::clamp(u.y, -1.0, 1.0)) / kDegree, std::atan2(u.z, u.x) / kDegree};
}

glm::dvec3 chartDirection(const ChartView& v, double x, double y)
{
  const double lat = (v.north + (v.south - v.north) * y / double(std::max(v.height, 1))) * kDegree;
  const double lon = (v.west + (v.east - v.west) * x / double(std::max(v.width, 1))) * kDegree;
  return {std::cos(lat) * std::cos(lon), std::sin(lat), std::cos(lat) * std::sin(lon)};
}

glm::dvec2 chartPoint(const ChartView& v, const glm::dvec3& d)
{
  const glm::dvec2 ll  = latitudeLongitudeOf(d);
  double           lon = ll.y;
  const double     mid = middleLongitude(v);
  lon += 360.0 * std::round((mid - lon) / 360.0);
  return {(lon - v.west) / (v.east - v.west) * double(v.width), (ll.x - v.north) / (v.south - v.north) * double(v.height)};
}

double chartPixel(const World& w, const ChartView& v)
{
  return (v.north - v.south) * kDegree * w.spec.radius / double(std::max(v.height, 1));
}

std::vector<uint8_t> chartImage(const World& w, const ChartView& v, const std::atomic<bool>* stop)
{
  const int n = std::max(v.width, 1), m = std::max(v.height, 1);
  const auto stopped = [&] { return stop && stop->load(); };
  // the ground at each pixel's middle, read as finely as a pixel
  const double        dy = chartPixel(w, v);
  std::vector<Sample> samples(size_t(n) * size_t(m));
  fv::jobs::parallelFor(samples.size(), 256, [&](size_t begin, size_t end) {
    for(size_t k = begin; k < end && !stopped(); k++)
      samples[k] = sample(w, chartDirection(v, double(k % size_t(n)) + 0.5, double(k / size_t(n)) + 0.5), dy);
  });
  if(stopped())
    return {};

  // its colour, shaded by the slope between its neighbours
  std::vector<glm::dvec3> colour(samples.size());
  const auto              height = [&](int i, int j) {
    const Sample& s = samples[size_t(std::clamp(j, 0, m - 1)) * size_t(n) + size_t(std::clamp(i, 0, n - 1))];
    return std::max(s.elevation, 0.0);
  };
  fv::jobs::parallelFor(size_t(m), 8, [&](size_t begin, size_t end) {
    for(size_t row = begin; row < end; row++)
    {
      const int    j  = int(row);
      const double dx = std::max(dy * (v.east - v.west) / (v.north - v.south) * double(m) / double(n)
                                     * std::cos((v.north + (v.south - v.north) * (double(j) + 0.5) / double(m)) * kDegree),
                                 1e-3);
      for(int i = 0; i < n; i++)
      {
        const Sample&    s      = samples[size_t(j) * size_t(n) + size_t(i)];
        const glm::dvec3 normal = glm::normalize(glm::dvec3(-(height(i + 1, j) - height(i - 1, j)) / (2.0 * dx),
                                                            (height(i, j + 1) - height(i, j - 1)) / (2.0 * dy), 1.0));
        glm::dvec3       c;
        if(s.elevation < 0.0)
          c = glm::mix(kShallow, kDeep, std::clamp(-s.elevation / kDeepAt, 0.0, 1.0));
        else if(s.water > s.elevation)
          c = kInland;
        else
        {
          const double lit = 0.25 + 1.1 * std::max(glm::dot(normal, kLight), 0.0);
          c = glm::dvec3(surfaceColour(w, s, float(std::sqrt(std::max(1.0 - normal.z * normal.z, 0.0)) / normal.z))) * lit * 2.5;
        }
        colour[size_t(j) * size_t(n) + size_t(i)] = c;
      }
    }
  });

  // lines over it: a curve drawn as discs along it, each at least half a pixel across
  const auto line = [&](auto at, auto metres, const glm::dvec3& c) {
    const glm::dvec2 a = chartPoint(v, at(0.0)), b = chartPoint(v, at(0.5)), e = chartPoint(v, at(1.0));
    const double     margin = 4.0;
    if(std::max({a.x, b.x, e.x}) < -margin || std::min({a.x, b.x, e.x}) > n + margin || std::max({a.y, b.y, e.y}) < -margin
       || std::min({a.y, b.y, e.y}) > m + margin)
      return;
    const int steps = std::clamp(int(2.0 * (glm::length(b - a) + glm::length(e - b))) + 1, 1, 4096);
    for(int k = 0; k <= steps; k++)
    {
      const double     t = double(k) / double(steps);
      const glm::dvec2 p = chartPoint(v, at(t));
      const double     r = std::max(0.5, 0.5 * metres(t) / dy);
      for(int j = int(std::floor(p.y - r)); j <= int(std::floor(p.y + r)); j++)
        for(int i = int(std::floor(p.x - r)); i <= int(std::floor(p.x + r)); i++)
          if(i >= 0 && j >= 0 && i < n && j < m && (glm::length(glm::dvec2(i + 0.5, j + 0.5) - p) <= r + 0.5))
            colour[size_t(j) * size_t(n) + size_t(i)] = c;
    }
  };
  const double R = w.spec.radius;
  if(v.rivers && w.base)
    for(const RiverPiece& p : w.base->pieces)
    {
      if(stopped())
        return {};
      line([&](double t) { return riverMiddleAt(p, t, R); },
           [&](double t) {
             const double s = 1.0 - t;
             return 2.0 * (double(p.half[0]) * s * s + double(p.half[1]) * 2.0 * s * t + double(p.half[2]) * t * t);
           },
           kRiver);
    }
  if(v.roads && w.places && w.places->catalogue)
    for(const RoadPiece& p : w.places->roads)
    {
      if(stopped())
        return {};
      const RoadKind&  kind = w.places->catalogue->roads[std::min<size_t>(p.kind, w.places->catalogue->roads.size() - 1)];
      const glm::dvec3 c    = glm::dvec3(kind.colour) * (kind.earth ? 1.6 : 1.0);
      line([&](double t) {
             const double s = 1.0 - t;
             return glm::normalize(p.a * (s * s) + p.b * (2.0 * s * t) + p.c * (t * t));
           },
           [&](double) { return 2.0 * double(p.half); }, c);
    }

  std::vector<uint8_t> out(samples.size() * 3);
  for(size_t k = 0; k < colour.size(); k++)
    for(int c = 0; c < 3; c++)
      out[k * 3 + size_t(c)] = uint8_t(255.0 * std::pow(std::clamp(colour[k][c], 0.0, 1.0), 1.0 / 2.2) + 0.5);
  return out;
}

}  // namespace fv::world
