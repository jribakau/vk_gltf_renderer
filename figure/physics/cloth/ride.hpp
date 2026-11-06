#pragma once

// Live cloth between its steps (physics-plan 13c). A live step's drape is drawn for a few frames after
// the pose it was simulated at, by the garment's own skin, and a dress's weights, spread over both
// thighs and the hip, don't follow a thigh that swings forward: the body came through the dress until
// the next step landed (10TEST's walk, the Charming Witch Dress settled and then drawn 50 to 100 ms on:
// 800 to 1,250 of its vertices inside the body, against 36 to 63 settled). What lies near the body rides
// it instead, each vertex on its nearest face, carried with the face to the pose shown.

#include <cstdint>
#include <vector>

#include <figure/physics/cloth/surface.hpp>

namespace fv::physics::cloth {

// A vertex within kRideNear cm of the body rides it whole, less further off, and past kRideFar only its
// own skin carries it. Measured on 10TEST's walk (fv_parity, the dress drawn 50 and 100 ms after it was
// settled): riding within 1 cm, gone by 3, left 139-331 vertices over 3 mm inside 100 ms on; 3 to 8
// left 0-14; past that a vertex between the legs rode the other leg.
constexpr float kRideNear = 3.f;
constexpr float kRideFar  = 8.f;

// Some of a drape's vertices, each on a face of the body.
struct Ride
{
  std::vector<uint32_t> vertices;  // the drape's vertices that ride, ascending
  std::vector<uint32_t> face;      // per riding vertex: its face, of the body's (an index into its tris / 3)
  std::vector<uint32_t> corners;   // the body's vertices their faces have, ascending, once each
  std::vector<uint32_t> at;        // per riding vertex: its face's three corners, as indices into `corners`
  std::vector<float>    bary;      // per riding vertex: where on its face, a weight per corner
  std::vector<float>    local;     // per riding vertex: its offset from there in the face's frame
  std::vector<float>    weight;    // per riding vertex: how much it rides, in (0, 1] (0: Live's proxy holds it, it keeps its skin)

  bool empty() const { return vertices.empty(); }
};

// A face's point at `bary` and its frame there: e1 along its first side, n its normal (its winding's,
// outward on the body: its faces wind as keepOut and the collisions read them), e2 across. A face with no
// area has none (`ok` false), and an offset on it is taken as it is.
struct RideFrame
{
  double q[3], e1[3], e2[3], n[3];
  bool   ok = false;
};
RideFrame rideFrame(const float* a, const float* b, const float* c, const float* bary);

// `drape` (xyz per vertex, in the job's space) on the nearest faces of `body` as it lies at `bodyAt`, all
// but what is `held` to its skin (per vertex, 1: a vertex of no strength, or one the cloth leaves out).
// With `hint`, the last ride of the same drape on the same body, each vertex it has is found again on its
// face there and no other is looked for: the bind between full ones, a test a vertex for a search.
Ride bindRide(const Surface& body, const std::vector<float>& bodyAt, const std::vector<float>& drape, const std::vector<uint8_t>& held,
              const Ride* hint = nullptr, float nearCm = kRideNear, float farCm = kRideFar);

// The ride's corners where `skin` (the body's, at a pose) has them: xyz per entry of ride.corners.
std::vector<float> rideCorners(const Ride& ride, const SurfaceSkin& skin);

// Each riding vertex where its face carries it, the face's corners at `corners` (rideCorners'), from
// offsets `local` (the ride's own, or a set rideLocal made): xyz per riding vertex.
std::vector<float> rideAt(const Ride& ride, const std::vector<float>& corners, const std::vector<float>& local);

// The offsets on the ride's faces, their corners at `corners`, that rideAt turns back into `points` (xyz
// per riding vertex) there.
std::vector<float> rideLocal(const Ride& ride, const std::vector<float>& corners, const std::vector<float>& points);

}  // namespace fv::physics::cloth
