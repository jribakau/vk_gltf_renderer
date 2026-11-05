#include <figure/physics/cloth/surface.hpp>

#include <algorithm>

#include <figure/jobs/pool.hpp>
#include <figure/physics/cloth/geometry.hpp>

namespace fv::physics::cloth {

void skin(const Surface& m, const std::vector<Mat4>& bones, float* out)
{
  const size_t n = m.vertexCount();
  if(!m.skinned() || bones.size() != m.bones.size())
  {
    std::copy(m.posed.begin(), m.posed.end(), out);
    return;
  }
  std::vector<Mat34> at(bones.size());
  for(size_t b = 0; b < bones.size(); b++)
    at[b] = Mat34::of(m.toSpace * bones[b]);
  // on the pool, each vertex its own: the same numbers on any number of threads (a body and a dress are
  // 67k vertices, skinned twice a frame for the GPU solver)
  jobs::parallelFor(n, 2048, [&](size_t begin, size_t end) {
    for(size_t v = begin; v < end; v++)
    {
      float blend[12] = {};
      float wsum      = 0;
      for(size_t j = 0; j < 4; j++)
      {
        const float  w    = m.skinWeight[v * 4 + j];
        const size_t bone = m.skinIndex[v * 4 + j];
        if(w == 0 || bone >= at.size())
          continue;
        wsum += w;
        for(int c = 0; c < 12; c++)
          blend[c] += at[bone].m[c] * w;
      }
      const float px = m.restLocal[v * 3], py = m.restLocal[v * 3 + 1], pz = m.restLocal[v * 3 + 2];
      if(wsum <= 0)
      {
        const Vec3 p = m.toSpace.transformPoint({px, py, pz});
        out[v * 3]     = float(p.x);
        out[v * 3 + 1] = float(p.y);
        out[v * 3 + 2] = float(p.z);
        continue;
      }
      out[v * 3]     = blend[0] * px + blend[3] * py + blend[6] * pz + blend[9];
      out[v * 3 + 1] = blend[1] * px + blend[4] * py + blend[7] * pz + blend[10];
      out[v * 3 + 2] = blend[2] * px + blend[5] * py + blend[8] * pz + blend[11];
    }
  });
}

const char* poseFreeDifference(const Surface& a, const Surface& b, bool colliderRest)
{
#define FV_DIFFERS(field)                                                                                                    \
  if(!(a.field == b.field))                                                                                                  \
    return #field;
  FV_DIFFERS(kind)
  const bool restIsPose = colliderRest && a.kind == Surface::Kind::eCollider;
  if(restIsPose)
  {
    FV_DIFFERS(rest.size())
    FV_DIFFERS(restLocal.size())
  }
  else
  {
    FV_DIFFERS(rest)
    FV_DIFFERS(restLocal)
  }
  FV_DIFFERS(posed.size())
  FV_DIFFERS(tris)
  FV_DIFFERS(polyStart)
  FV_DIFFERS(polyVerts)
  FV_DIFFERS(strength)
  FV_DIFFERS(stretch)
  FV_DIFFERS(shear)
  FV_DIFFERS(bend)
  FV_DIFFERS(buckling)
  FV_DIFFERS(bucklingRatio)
  FV_DIFFERS(density)
  FV_DIFFERS(contraction)
  FV_DIFFERS(damping)
  FV_DIFFERS(friction)
  FV_DIFFERS(offset)
  FV_DIFFERS(layer)
  FV_DIFFERS(selfCollide)
  FV_DIFFERS(collide)
  FV_DIFFERS(visible)
  FV_DIFFERS(skinIndex)
  FV_DIFFERS(skinWeight)
  FV_DIFFERS(bones.size())
  FV_DIFFERS(toSpace.e)
#undef FV_DIFFERS
  return nullptr;
}

bool sameButPose(const Surface& a, const Surface& b)
{
  return poseFreeDifference(a, b) == nullptr;
}

Surface Surface::collider(std::vector<float> rest, std::vector<float> posed, std::vector<uint32_t> tris, int layer, float offset,
                          float friction, const std::vector<uint8_t>& collide)
{
  Surface m;
  const size_t n = rest.size() / 3;
  m.rest         = std::move(rest);
  m.posed        = std::move(posed);
  m.tris         = std::move(tris);
  m.strength.assign(n, 0.f);
  m.stretch.assign(n, 0.8f);
  m.shear.assign(n, 0.2f);
  m.bend.assign(n, 0.5f);
  m.buckling.assign(n, 0.05f);
  m.bucklingRatio.assign(n, 0.7f);
  m.density.assign(n, 180.f);
  m.contraction.assign(n, 1.f);
  m.damping.assign(n, 0.1f);
  m.friction.assign(n, friction);
  m.offset.assign(n, offset);
  m.layer.assign(n, int16_t(layer));
  m.selfCollide.assign(n, 0);
  m.collide = collide.size() == n ? collide : std::vector<uint8_t>(n, 1);
  m.visible.assign(n, 1);
  return m;
}

}  // namespace fv::physics::cloth
