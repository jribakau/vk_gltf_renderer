#include <figure/physics/hair/strands.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>

#include <figure/jobs/pool.hpp>
#include <figure/physics/collide/distance_field.hpp>
#include <figure/physics/p3.hpp>
#include <figure/physics/particles/particles.hpp>

namespace fv::physics::hair {

namespace {

// The files' shape stiffnesses are per their own solver's step, in no stated unit: a pull of this share
// of them per step leaves hair hanging with its authored bends showing. The first span takes its
// stiffness at this many times over, so that hair leaves the scalp the way it grows before gravity has it.
constexpr float kShapeScale = 0.25f;
constexpr float kRootShape  = 12.f;
// Velocity kept per step: a settle wants rest, not the swing of real hair.
constexpr float kKeepVelocity = 0.95f;
// The share of a point's correction by the next one that its velocity takes (the paper's s_damp).
constexpr float kFollowDamping = 0.9f;
// A point moves no further per step, so it can't cross a collider between two looks.
constexpr float kMaxMove = 1.5f;
// At Volume 1, on top of a strand's offset: how much further off a collider its root stands, cm, and
// as a multiple of the offset (a surface the file sets further out goes further out still)
constexpr float kVolumeLift = 4.f, kVolumeOffsets = 2.f;
// At Volume 1: the pull towards the way a strand grows at its root, per step, and how far along the
// strand that has faded to nothing, as a share of its length
constexpr float kVolumeHold = 0.15f, kVolumeHoldReach = 0.35f;
// Styled hair: how much of a span's direction is its authored one each step at its root, from Volume 0
// to 1, and what share of that is left at the tip, which droops most. Against gravity's 0.09 rad a step
// on a 3 cm span. Its lift at Volume 1, cm: it already stands where its author put it.
constexpr float kStyleLimp = 0.15f, kStyleFull = 0.9f, kStyleTip = 0.4f, kStyleLift = 1.5f;
// Frames a strand must lie still before it stops being solved
constexpr int kStillFrames = 2;
// Under this share of strands still moving, the hair counts as at rest
constexpr float kAwakeShare = 0.005f;

// One strand's simulated points: `at` indexes into the strand's own points
struct Chain
{
  std::vector<int> at;
  std::vector<P3>  x, prev, shape, shapeUnit;  // shape: each span's authored vector, x[i + 1] - x[i], as posed
  std::vector<float> length;
  // frames in a row it has barely moved
  int still = 0;
};

Chain chainOf(const StrandSet& s, size_t strand, float spacing)
{
  Chain      c;
  const int  begin = s.start[strand], n = s.start[strand + 1] - begin;
  const auto point = [&](int k) { return pointAt(s.posed, s.verts[size_t(begin + k)]); };
  c.at.push_back(0);
  float run = 0;
  for(int k = 1; k < n; k++)
  {
    run += len(point(k) - point(k - 1));
    if(run >= spacing || k == n - 1)
    {
      c.at.push_back(k);
      run = 0;
    }
  }
  c.x.reserve(c.at.size());
  for(int k : c.at)
    c.x.push_back(point(k));
  c.prev = c.x;
  for(size_t i = 0; i + 1 < c.x.size(); i++)
  {
    c.shape.push_back(c.x[i + 1] - c.x[i]);
    c.shapeUnit.push_back(unit(c.shape.back()));
    // the chord, as the waves laid back on it hold the rest of the strand's length
    c.length.push_back(len(c.shape.back()));
  }
  return c;
}

}  // namespace

std::vector<float> settle(const StrandSet& s, const std::vector<cloth::Surface>& colliders, const StrandOptions& options)
{
  std::vector<float> out = s.posed;
  const size_t       strands = s.start.empty() ? 0 : s.start.size() - 1;
  if(!strands)
    return out;

  std::vector<Chain> chains(strands);
  jobs::parallelFor(strands, 64, [&](size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++)
      if(s.start[i + 1] - s.start[i] >= 2 && s.strength[i] > 0)
        chains[i] = chainOf(s, i, options.spacing);
  });

  const auto liftOf = [&](size_t strand) { return s.volume * (s.styled ? kStyleLift : kVolumeLift + kVolumeOffsets * s.offset[strand]); };

  // The field needs only what the hair can reach: its roots, out to its longest strand.
  float offset = 0, longest = 0;
  P3    lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
  for(size_t i = 0; i < strands; i++)
  {
    const Chain& c = chains[i];
    if(c.x.size() < 2 || !s.collide[i])
      continue;
    float length = 0;
    for(const float l : c.length)
      length += l;
    longest = std::max(longest, length);
    offset  = std::max(offset, s.offset[i] + liftOf(i));
    lo      = {std::min(lo.x, c.x[0].x), std::min(lo.y, c.x[0].y), std::min(lo.z, c.x[0].z)};
    hi      = {std::max(hi.x, c.x[0].x), std::max(hi.y, c.x[0].y), std::max(hi.z, c.x[0].z)};
  }
  const DistanceField field(longest > 0 ? colliders : std::vector<cloth::Surface>{}, offset, lo - P3{longest, longest, longest}, hi + P3{longest, longest, longest});

  const float h = 1.f / 30.f / float(std::max(1, options.substeps));
  const P3    g{s.gravity[0] * h * h, s.gravity[1] * h * h, s.gravity[2] * h * h};

  const auto step = [&](size_t strand) {
    Chain&       c = chains[strand];
    const size_t n = c.x.size();
    const float  shapeRoot = s.shapeRoot[strand] * kShapeScale, shapeTip = s.shapeTip[strand] * kShapeScale;
    const float  lift      = liftOf(strand);
    const float  style     = s.styled ? mix(kStyleLimp, kStyleFull, s.volume) : 0.f;
    const bool   collide   = s.collide[strand] != 0;
    const float  friction  = std::clamp(s.friction[strand], 0.f, 1.f);
    // where each point goes: its motion, damped, and gravity
    thread_local std::vector<P3> pred, correction;
    pred.resize(n);
    correction.resize(n);
    pred[0] = c.x[0];
    for(size_t i = 1; i < n; i++)
    {
      P3 move = (c.x[i] - c.prev[i]) * kKeepVelocity + g;
      if(const float l = len(move); l > kMaxMove)
        move = move * (kMaxMove / l);
      pred[i] = c.x[i] + move;
    }
    // Root to tip, each span back to its length, leaning towards its authored bend from the span above,
    // and away from the colliders, the correction kept for the velocity. Every point builds on the one
    // before it where that one came to rest, so one pass holds both.
    P3 parentNow = c.shapeUnit[0], parentWas = parentNow;
    for(size_t i = 1; i < n; i++)
    {
      const P3    was       = pred[i];
      const P3    shapeUnit = c.shapeUnit[i - 1];
      const P3    authored  = i == 1 ? shapeUnit : unit(turned(shapeUnit, parentWas, parentNow), shapeUnit);
      const float along     = float(i - 1) / float(std::max<size_t>(1, n - 2));
      const float hold      = s.styled ? 0.f : s.volume * kVolumeHold * std::max(0.f, 1 - along / kVolumeHoldReach);
      const float pull      = std::min(1.f, std::max(hold, (i == 1 ? kRootShape : 1.f) * mix(shapeRoot, shapeTip, along)));
      const P3    now       = unit(pred[i] - pred[i - 1], authored);
      P3          dir       = unit(now * (1 - pull) + authored * pull, now);
      if(style > 0)
      {
        const float keep = style * mix(1.f, kStyleTip, along);
        dir              = unit(dir * (1 - keep) + shapeUnit * keep, dir);
      }
      particles::projectLeader(pred[i - 1], pred[i], dir, c.length[i - 1]);
      P3 normal;
      if(collide && field.pushOut(pred[i], s.offset[strand] + lift * (1 - along * along), normal))
      {
        // loses motion along what it rests on, and stays its length from the point above
        const P3 move    = pred[i] - c.x[i];
        const P3 tangent = move - normal * dot(move, normal);
        pred[i]          = pred[i] - tangent * friction;
        dir              = unit(pred[i] - pred[i - 1], dir);
        particles::projectLeader(pred[i - 1], pred[i], dir, c.length[i - 1]);
      }
      correction[i] = pred[i] - was;
      parentWas     = shapeUnit;
      parentNow     = dir;
    }
    // the velocity next time: a point's own move, without most of what holding the next point to its
    // length will have asked of it, which would otherwise build up along the strand
    for(size_t i = 1; i < n; i++)
    {
      c.prev[i] = i + 1 < n ? c.x[i] + correction[i + 1] * kFollowDamping : c.x[i];
      c.x[i]    = pred[i];
    }
  };

  for(int frame = 0; frame < options.maxFrames; frame++)
  {
    if(options.cancelled && options.cancelled())
      break;
    std::atomic<size_t> awake{0};
    jobs::parallelFor(strands, 32, [&](size_t begin, size_t end) {
      size_t          moving = 0;
      std::vector<P3> was;
      for(size_t i = begin; i < end; i++)
      {
        Chain& c = chains[i];
        if(c.x.size() < 2 || c.still >= kStillFrames)
          continue;
        was = c.x;
        for(int k = 0; k < options.substeps; k++)
          step(i);
        float most = 0;
        for(size_t p = 0; p < was.size(); p++)
          most = std::max(most, len(c.x[p] - was[p]));
        // a frame or two in, once it has had the chance to start moving
        c.still = frame > 3 && most < options.restMove ? c.still + 1 : 0;
        if(c.still < kStillFrames)
          moving++;
      }
      awake += moving;
    });
    if(frame > 3 && float(awake.load()) <= float(strands) * kAwakeShare)
      break;
  }

  // Every point of each strand: along its span as moved, its offset from the span turned as the span
  // turned, then as far from the posed shape as the strand's strength says.
  jobs::parallelFor(strands, 64, [&](size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++)
    {
      const Chain& c = chains[i];
      if(c.x.size() < 2)
        continue;
      const int   first    = s.start[i];
      const float strength = std::min(1.f, s.strength[i]);
      const auto  posedAt  = [&](int k) { return pointAt(s.posed, s.verts[size_t(first + k)]); };
      for(size_t span = 0; span + 1 < c.x.size(); span++)
      {
        const int a = c.at[span], b = c.at[span + 1];
        const P3  from = c.shapeUnit[span], to = unit(c.x[span + 1] - c.x[span], from);
        const P3  pa = posedAt(a);
        for(int k = a; k <= b; k++)
        {
          const P3    was    = posedAt(k);
          const float t      = dot(was - pa, from) / std::max(1e-5f, len(c.shape[span]));
          const P3    onSpan = pa + c.shape[span] * t;
          const P3    now    = c.x[span] + (c.x[span + 1] - c.x[span]) * t + turned(was - onSpan, from, to);
          const P3    final  = was + (now - was) * strength;
          const int   v      = s.verts[size_t(first + k)];
          out[size_t(v) * 3]     = final.x;
          out[size_t(v) * 3 + 1] = final.y;
          out[size_t(v) * 3 + 2] = final.z;
        }
      }
    }
  });
  return out;
}

}  // namespace fv::physics::hair
