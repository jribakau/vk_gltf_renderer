#include <figure/wardrobe/rigid_follow.hpp>

#include <algorithm>

#include <figure/garment/rigid.hpp>

namespace fv::wardrobe {

Mat4 followWorld(const Mat4& meshWorld, const Mat4& restInMesh, std::span<const float> rest, std::span<const float> now, bool turns)
{
  const size_t      n = std::min(rest.size(), now.size()) / 3;
  std::vector<Vec3> from(n), to(n);
  for(size_t i = 0; i < n; i++)
  {
    from[i] = {rest[i * 3], rest[i * 3 + 1], rest[i * 3 + 2]};
    to[i]   = {now[i * 3], now[i * 3 + 1], now[i * 3 + 2]};
  }
  // any turn: a sleeve's button turns with the arm
  std::optional<garment::RigidMove> move = garment::bestMove(from, to, 180);
  if(!move)
    return meshWorld * restInMesh;
  if(!turns)
    move->turn = Quat{};
  return meshWorld * move->matrix() * restInMesh;
}

void setTransform(Node& node, const Mat4& m)
{
  node.position   = {m.e[12], m.e[13], m.e[14]};
  node.quaternion = rotationOf(m);
  const Vec3   c0{m.e[0], m.e[1], m.e[2]}, c1{m.e[4], m.e[5], m.e[6]}, c2{m.e[8], m.e[9], m.e[10]};
  const double det = c0.x * (c1.y * c2.z - c1.z * c2.y) - c0.y * (c1.x * c2.z - c1.z * c2.x) + c0.z * (c1.x * c2.y - c1.y * c2.x);
  node.scale       = {det < 0 ? -length(c0) : length(c0), length(c1), length(c2)};
}

}  // namespace fv::wardrobe
