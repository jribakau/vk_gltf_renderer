#include "fv_world_sea.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>

namespace fv::world {

namespace {

constexpr float kTau = 6.28318530717958F;

uint32_t mix(uint32_t v)
{
  v = v * 747796405U + 2891336453U;
  const uint32_t w = ((v >> ((v >> 28U) + 4U)) ^ v) * 277803737U;
  return (w >> 22U) ^ w;
}

struct Rng
{
  uint32_t state;
  float    next()
  {
    state = mix(state + 0x9E3779B9U);
    return float(state >> 8U) * (1.0F / 16777216.0F);
  }
  float range(float a, float b) { return a + (b - a) * next(); }
};

// Smooth value noise on a unit lattice, 0 to 1, and two octaves of it
float lattice(const glm::vec3& p, uint32_t seed)
{
  const glm::vec3 f = glm::floor(p);
  const glm::vec3 t = p - f;
  const glm::vec3 u = t * t * (3.0F - 2.0F * t);
  const auto      at = [&](int dx, int dy, int dz) {
    return float(mix(mix(mix(seed + uint32_t(int(f.x) + dx)) + uint32_t(int(f.y) + dy)) + uint32_t(int(f.z) + dz)) >> 8U) * (1.0F / 16777216.0F);
  };
  const float x00 = glm::mix(at(0, 0, 0), at(1, 0, 0), u.x), x10 = glm::mix(at(0, 1, 0), at(1, 1, 0), u.x);
  const float x01 = glm::mix(at(0, 0, 1), at(1, 0, 1), u.x), x11 = glm::mix(at(0, 1, 1), at(1, 1, 1), u.x);
  return glm::mix(glm::mix(x00, x10, u.y), glm::mix(x01, x11, u.y), u.z);
}
float lumps(const glm::vec3& p, uint32_t seed)
{
  return (2.0F * lattice(p, seed) + lattice(p * 2.0F + glm::vec3(17.3F, -9.1F, 5.7F), seed + 1U)) / 3.0F;
}

// A mesh being made
struct Make
{
  SeaMesh& m;

  uint32_t vertex(const glm::vec3& p, const glm::vec3& colour)
  {
    m.positions.push_back(p);
    m.colors.emplace_back(colour, 1.0F);
    return uint32_t(m.positions.size() - 1);
  }
  void triangle(uint32_t a, uint32_t b, uint32_t c) { m.indices.insert(m.indices.end(), {a, b, c}); }
  void quad(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
  {
    triangle(a, b, c);
    triangle(a, c, d);
  }

  // A tube along `points`, `segments` round, its radius and colour by the point; closed at its far
  // end by a point `cap` radii further on (0: left open). Returns its last ring's first vertex.
  uint32_t tube(const std::vector<glm::vec3>& points, const std::function<float(size_t)>& radius, uint32_t segments,
                const std::function<glm::vec3(size_t)>& colour, float cap = 0.0F)
  {
    uint32_t  last = 0;
    glm::vec3 side(1.0F, 0.0F, 0.0F);
    for(size_t k = 0; k < points.size(); k++)
    {
      const glm::vec3 along = glm::normalize(points[std::min(k + 1, points.size() - 1)] - points[k > 0 ? k - 1 : 0]);
      // a side carried along the tube, so its rings don't twist
      side = side - along * glm::dot(side, along);
      if(glm::dot(side, side) < 1e-6F)
        side = glm::cross(along, std::abs(along.y) < 0.9F ? glm::vec3(0.0F, 1.0F, 0.0F) : glm::vec3(1.0F, 0.0F, 0.0F));
      side                  = glm::normalize(side);
      const glm::vec3 other = glm::cross(along, side);
      const uint32_t  ring  = uint32_t(m.positions.size());
      for(uint32_t s = 0; s < segments; s++)
      {
        const float a = kTau * float(s) / float(segments);
        vertex(points[k] + (side * std::cos(a) + other * std::sin(a)) * radius(k), colour(k));
      }
      if(k > 0)
        for(uint32_t s = 0; s < segments; s++)
        {
          const uint32_t s2 = (s + 1) % segments;
          quad(last + s, last + s2, ring + s2, ring + s);
        }
      last = ring;
    }
    if(cap != 0.0F && points.size() > 1)
    {
      const size_t    n     = points.size() - 1;
      const glm::vec3 along = glm::normalize(points[n] - points[n - 1]);
      const uint32_t  tip   = vertex(points[n] + along * (cap * radius(n)), colour(n));
      for(uint32_t s = 0; s < segments; s++)
        triangle(last + s, last + (s + 1) % segments, tip);
    }
    return last;
  }

  // A ribbon along `points`, `width` across at each, lying across `across`
  void ribbon(const std::vector<glm::vec3>& points, const std::function<float(size_t)>& width, const glm::vec3& across,
              const std::function<glm::vec3(size_t)>& colour)
  {
    uint32_t last = 0;
    for(size_t k = 0; k < points.size(); k++)
    {
      const glm::vec3 half = across * (0.5F * width(k));
      const uint32_t  a    = vertex(points[k] - half, colour(k));
      vertex(points[k] + half, colour(k));
      if(k > 0)
        quad(last, last + 1, a + 1, a);
      last = a;
    }
  }

  // Smooth normals from the triangles, and how high it stands
  void finish()
  {
    m.normals.assign(m.positions.size(), glm::vec3(0.0F));
    for(size_t t = 0; t + 2 < m.indices.size(); t += 3)
    {
      const uint32_t  a = m.indices[t], b = m.indices[t + 1], c = m.indices[t + 2];
      const glm::vec3 n = glm::cross(m.positions[b] - m.positions[a], m.positions[c] - m.positions[a]);
      m.normals[a] += n;
      m.normals[b] += n;
      m.normals[c] += n;
    }
    m.height = 0.0F;
    for(size_t v = 0; v < m.positions.size(); v++)
    {
      const float l = glm::length(m.normals[v]);
      m.normals[v]  = l > 1e-12F ? m.normals[v] / l : glm::vec3(0.0F, 1.0F, 0.0F);
      m.height      = std::max(m.height, m.positions[v].y);
    }
  }
};

// A patch of a seagrass meadow, a metre and a half across: a few tufts of ribbon leaves, each bent
// out from its tuft's middle
void seagrass(SeaMesh& m, Rng rng)
{
  Make            make{m};
  const glm::vec3 low(0.035F, 0.085F, 0.02F), high(0.09F, 0.2F, 0.05F);
  const int       tufts = 5 + int(rng.next() * 3.0F);
  for(int tuft = 0; tuft < tufts; tuft++)
  {
    const float     where = rng.next() * kTau, off = tuft == 0 ? 0.0F : rng.range(0.25F, 0.75F);
    const glm::vec3 foot(std::cos(where) * off, 0.0F, std::sin(where) * off);
    const int       blades = 9 + int(rng.next() * 6.0F);
    for(int i = 0; i < blades; i++)
    {
      const float     a = rng.next() * kTau, from = rng.range(0.0F, 0.13F), tall = rng.range(0.3F, 0.75F), bend = rng.range(0.05F, 0.35F) * tall;
      const glm::vec3 out(std::cos(a), 0.0F, std::sin(a));
      const glm::vec3 across = glm::normalize(glm::vec3(-out.z, 0.0F, out.x) + out * rng.range(-0.4F, 0.4F));
      const float     wide   = rng.range(0.014F, 0.026F);
      std::vector<glm::vec3> points;
      for(int k = 0; k <= 4; k++)
      {
        const float t = float(k) / 4.0F;
        points.push_back(foot + out * (from + bend * t * t) + glm::vec3(0.0F, tall * t, 0.0F));
      }
      make.ribbon(points, [&](size_t k) { return wide * (1.0F - 0.85F * float(k) / 4.0F); }, across,
                  [&](size_t k) { return glm::mix(low, high, float(k) / 4.0F); });
    }
  }
  m.leaves = true;
  make.finish();
}

// A stalk that sways up through the water, with blades off it
void kelp(SeaMesh& m, Rng rng, float tall)
{
  Make            make{m};
  const glm::vec3 stem(0.07F, 0.075F, 0.02F), leaf(0.085F, 0.1F, 0.022F), tip(0.16F, 0.15F, 0.04F);
  const float     swayA = rng.range(0.15F, 0.4F), swayB = rng.range(0.1F, 0.3F), phaseA = rng.next() * kTau, phaseB = rng.next() * kTau;
  const int       steps = 16;
  std::vector<glm::vec3> stalk;
  for(int k = 0; k <= steps; k++)
  {
    const float t = float(k) / float(steps);
    stalk.emplace_back(swayA * t * std::sin(phaseA + 4.0F * t), tall * t, swayB * t * std::sin(phaseB + 5.5F * t));
  }
  make.tube(stalk, [&](size_t k) { return 0.035F * (1.0F - 0.6F * float(k) / float(steps)); }, 5, [&](size_t) { return stem; }, 1.0F);
  const int blades = int(tall * 6.0F);
  for(int i = 0; i < blades; i++)
  {
    const float     t  = 0.12F + 0.88F * (float(i) + rng.next()) / float(blades);
    const float     at = t * float(steps);
    const int       k  = std::min(int(at), steps - 1);
    const glm::vec3 from = glm::mix(stalk[size_t(k)], stalk[size_t(k) + 1], at - float(k));
    const float     a = rng.next() * kTau, reach = rng.range(0.9F, 1.8F), wide = rng.range(0.18F, 0.32F);
    const glm::vec3 out(std::cos(a), 0.0F, std::sin(a));
    const glm::vec3 across(-out.z, 0.0F, out.x);
    std::vector<glm::vec3> points;
    for(int j = 0; j <= 4; j++)
    {
      // out and a little up, then drooping
      const float s = float(j) / 4.0F;
      points.push_back(from + out * (reach * s) + glm::vec3(0.0F, reach * (0.45F * s - 0.55F * s * s), 0.0F));
    }
    make.ribbon(points, [&](size_t j) { return wide * (j == 0 ? 0.25F : j == 4 ? 0.3F : 1.0F); }, across,
                [&](size_t j) { return glm::mix(leaf, tip, float(j) / 4.0F); });
  }
  m.leaves = true;
  make.finish();
}

// Branches that fork: a bush, or a fan (flat in z)
void coral(SeaMesh& m, Rng rng, bool fan, const glm::vec3& colour)
{
  Make            make{m};
  const glm::vec3 pale = glm::mix(colour, glm::vec3(0.75F, 0.7F, 0.62F), 0.55F);
  const std::function<void(glm::vec3, glm::vec3, float, float, int)> branch = [&](glm::vec3 p, glm::vec3 d, float r, float len, int depth) {
    std::vector<glm::vec3> points{p};
    for(int i = 0; i < 3; i++)
    {
      d = glm::normalize(d + glm::vec3(rng.range(-0.3F, 0.3F), rng.range(-0.1F, 0.3F), fan ? rng.range(-0.05F, 0.05F) : rng.range(-0.3F, 0.3F)));
      p += d * (len / 3.0F);
      points.push_back(p);
    }
    // darker in the bush's heart, pale at the tips
    const float deep = float(depth) / 3.0F;
    make.tube(points, [&](size_t k) { return r * (1.0F - 0.22F * float(k) / 3.0F); }, fan ? 5U : 6U,
              [&](size_t k) { return glm::mix(pale, colour * 0.75F, std::clamp(deep - float(k) / 9.0F, 0.0F, 1.0F)); }, depth == 0 ? 1.6F : 0.0F);
    if(depth > 0)
    {
      const int n = 2 + (rng.next() < 0.5F ? 1 : 0);
      for(int i = 0; i < n; i++)
        branch(p, glm::normalize(d + glm::vec3(rng.range(-0.8F, 0.8F), rng.range(-0.1F, 0.5F), fan ? rng.range(-0.08F, 0.08F) : rng.range(-0.8F, 0.8F))),
               r * 0.78F, len * 0.8F, depth - 1);
    }
  };
  const int trunks = fan ? 1 : 2 + int(rng.next() * 2.0F);
  for(int i = 0; i < trunks; i++)
    branch(glm::vec3(rng.range(-0.08F, 0.08F), 0.0F, rng.range(-0.08F, 0.08F)),
           glm::normalize(glm::vec3(rng.range(-0.5F, 0.5F), 1.0F, fan ? 0.0F : rng.range(-0.5F, 0.5F))), fan ? 0.06F : 0.08F, fan ? 0.5F : 0.45F, 3);
  make.finish();
}

// A profile turned about the vertical, with lumps: a vase, open at the top and dark inside
void vase(SeaMesh& m, Rng rng, const glm::vec3& colour)
{
  Make make{m};
  // radius, height, how dark (the last three are its inside, going down)
  const float    profile[][3] = {{0.10F, 0.0F, 0.6F},  {0.17F, 0.15F, 0.8F}, {0.26F, 0.45F, 1.0F}, {0.32F, 0.75F, 1.0F}, {0.30F, 0.92F, 1.0F},
                                 {0.24F, 0.9F, 0.45F}, {0.13F, 0.55F, 0.15F}, {0.02F, 0.3F, 0.05F}};
  const uint32_t seed = rng.state, segments = 14;
  const float    lump = rng.range(0.25F, 0.45F), ribs = float(3 + int(rng.next() * 4.0F));
  uint32_t       last = 0;
  for(size_t k = 0; k < std::size(profile); k++)
  {
    const float    r = profile[k][0], h = profile[k][1];
    const uint32_t ring = uint32_t(m.positions.size());
    for(uint32_t s = 0; s < segments; s++)
    {
      const float a  = kTau * float(s) / float(segments);
      const float rl = r * (1.0F + lump * (lumps(glm::vec3(std::cos(a) * 1.5F + h * 2.0F, h * 3.0F, std::sin(a) * 1.5F), seed) - 0.5F) + 0.07F * std::cos(ribs * a + h * 1.5F));
      make.vertex(glm::vec3(std::cos(a) * rl, h, std::sin(a) * rl), colour * profile[k][2]);
    }
    if(k > 0)
      for(uint32_t s = 0; s < segments; s++)
      {
        const uint32_t s2 = (s + 1) % segments;
        // (wound so its outside faces out: the rings go up the outside and down the inside)
        make.quad(last + s2, last + s, ring + s, ring + s2);
      }
    last = ring;
  }
  make.finish();
}

// A cluster of leaning tubes, each with a rim and a dark throat
void tubes(SeaMesh& m, Rng rng, const glm::vec3& colour)
{
  Make make{m};
  for(int i = 0; i < 9; i++)
  {
    const float a = rng.next() * kTau, from = rng.range(0.0F, 0.28F), tall = rng.range(0.45F, 1.1F), radius = rng.range(0.07F, 0.12F), lean = rng.range(0.05F, 0.3F);
    std::vector<glm::vec3> points;
    for(int k = 0; k <= 4; k++)
      points.emplace_back(std::cos(a) * (from + lean * float(k) / 4.0F * tall), tall * float(k) / 4.0F, std::sin(a) * (from + lean * float(k) / 4.0F * tall));
    const uint32_t rim = make.tube(points, [&](size_t k) { return radius * (0.8F + 0.25F * float(k) / 4.0F); }, 10,
                                   [&](size_t k) { return colour * (0.45F + 0.55F * float(k) / 4.0F); });
    const uint32_t throat = make.vertex(points[4] - glm::vec3(0.0F, radius * 1.2F, 0.0F), colour * 0.05F);
    for(uint32_t s = 0; s < 10; s++)
      make.triangle(rim + (s + 1) % 10, rim + s, throat);
  }
  make.finish();
}

// A lumpy ball cut by a few planes, a fifth of it in the ground
void rock(SeaMesh& m, Rng rng)
{
  Make make{m};
  // a ball of 320 triangles: twenty faces, each split in four, twice
  const float t = 0.5F * (1.0F + std::sqrt(5.0F));
  std::vector<glm::vec3> points = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  for(glm::vec3& p : points)
    p = glm::normalize(p);
  std::vector<uint32_t> faces = {0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
                                 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1};
  for(int pass = 0; pass < 2; pass++)
  {
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> middles;
    const auto middle = [&](uint32_t a, uint32_t b) {
      const auto key   = std::minmax(a, b);
      const auto found = middles.find(key);
      if(found != middles.end())
        return found->second;
      points.push_back(glm::normalize(points[a] + points[b]));
      return middles[key] = uint32_t(points.size() - 1);
    };
    std::vector<uint32_t> finer;
    for(size_t f = 0; f + 2 < faces.size(); f += 3)
    {
      const uint32_t a = faces[f], b = faces[f + 1], c = faces[f + 2], ab = middle(a, b), bc = middle(b, c), ca = middle(c, a);
      finer.insert(finer.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
    }
    faces.swap(finer);
  }
  const uint32_t seed = rng.state;
  glm::vec4      planes[4];
  for(glm::vec4& plane : planes)
    plane = glm::vec4(glm::normalize(glm::vec3(rng.range(-1.0F, 1.0F), rng.range(-1.0F, 1.0F), rng.range(-1.0F, 1.0F)) + glm::vec3(1e-3F)), rng.range(0.8F, 0.98F));
  const glm::vec3 grey(0.13F, 0.125F, 0.11F), brown(0.16F, 0.13F, 0.09F);
  float           lowest = 0.0F;
  for(glm::vec3& p : points)
  {
    const float r = 1.0F + 0.9F * (lumps(p * 1.6F, seed) - 0.5F);
    p             = glm::vec3(p.x * r, p.y * r * 0.8F, p.z * r);
    for(const glm::vec4& plane : planes)
      if(const float over = glm::dot(p, glm::vec3(plane)) - plane.w; over > 0.0F)
        p -= glm::vec3(plane) * over;
    lowest = std::min(lowest, p.y);
  }
  // half a metre across or so, standing with its lowest fifth under y = 0
  for(const glm::vec3& p : points)
    make.vertex((p - glm::vec3(0.0F, 0.8F * lowest, 0.0F)) * 0.5F, glm::mix(grey, brown, lumps(p * 3.0F, seed + 7U)) * (0.75F + 0.5F * lumps(p * 6.0F, seed + 9U)));
  m.indices = faces;
  make.finish();
}

}  // namespace

std::vector<SeaMesh> seaMeshes(int seed)
{
  std::vector<SeaMesh> out;
  const auto           add = [&](std::string_view role, uint32_t variant, const std::function<void(SeaMesh&, Rng)>& make) {
    SeaMesh m;
    m.role = uint8_t(std::find(kSeaRoles.begin(), kSeaRoles.end(), role) - kSeaRoles.begin());
    make(m, Rng{mix(uint32_t(seed) * 977U + uint32_t(m.role) * 131U + variant)});
    out.push_back(std::move(m));
  };
  for(uint32_t v = 0; v < 3; v++)
    add("seagrass", v, [](SeaMesh& m, Rng rng) { seagrass(m, rng); });
  for(uint32_t v = 0; v < 3; v++)
    add("kelp", v, [v](SeaMesh& m, Rng rng) { kelp(m, rng, 6.0F + 2.0F * float(v)); });
  const glm::vec3 corals[4] = {{0.55F, 0.18F, 0.22F}, {0.6F, 0.28F, 0.08F}, {0.32F, 0.16F, 0.46F}, {0.6F, 0.5F, 0.34F}};
  for(uint32_t v = 0; v < 4; v++)
    add("coral", v, [&, v](SeaMesh& m, Rng rng) { coral(m, rng, v % 2 == 1, corals[v]); });
  add("sponge", 0, [](SeaMesh& m, Rng rng) { vase(m, rng, glm::vec3(0.6F, 0.42F, 0.1F)); });
  add("sponge", 1, [](SeaMesh& m, Rng rng) { vase(m, rng, glm::vec3(0.45F, 0.16F, 0.08F)); });
  add("sponge", 2, [](SeaMesh& m, Rng rng) { tubes(m, rng, glm::vec3(0.55F, 0.3F, 0.38F)); });
  for(uint32_t v = 0; v < 3; v++)
    add("searock", v, [](SeaMesh& m, Rng rng) { rock(m, rng); });
  return out;
}

}  // namespace fv::world
