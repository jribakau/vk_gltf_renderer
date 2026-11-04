#include <figure/physics/rigid/world.hpp>

#include <algorithm>
#include <cmath>

namespace fv::physics {

namespace {

constexpr double kPi = 3.14159265358979323846;

// q turned by the small rotation vector r (world): q + 1/2 [r, 0] q, normalised
Quat turned(const Quat& q, const Vec3& r)
{
  const Quat d = Quat{r.x, r.y, r.z, 0} * q;
  return Quat{q.x + 0.5 * d.x, q.y + 0.5 * d.y, q.z + 0.5 * d.z, q.w + 0.5 * d.w}.normalized();
}

// q's rotation vector (axis times angle), the short way round
Vec3 rotationVector(Quat q)
{
  if(q.w < 0)
    q = {-q.x, -q.y, -q.z, -q.w};
  const double s = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
  if(s < 1e-12)
    return {2 * q.x, 2 * q.y, 2 * q.z};
  return Vec3{q.x, q.y, q.z} * (2 * std::atan2(s, q.w) / s);
}

Quat fromRotationVector(const Vec3& r)
{
  const double a = length(r);
  if(a < 1e-12)
    return Quat{r.x / 2, r.y / 2, r.z / 2, 1}.normalized();
  const double s = std::sin(a / 2) / a;
  return {r.x * s, r.y * s, r.z * s, std::cos(a / 2)};
}

double wrapPi(double a)
{
  return a - 2 * kPi * std::round(a / (2 * kPi));
}

Vec3 linearPart(const Mat4& m, const Vec3& v)
{
  return {m.e[0] * v.x + m.e[4] * v.y + m.e[8] * v.z, m.e[1] * v.x + m.e[5] * v.y + m.e[9] * v.z, m.e[2] * v.x + m.e[6] * v.y + m.e[10] * v.z};
}


// A block of the tree solve, up to 6 by 6.
struct Blk
{
  int    r = 0, c = 0;
  double a[6][6]{};
};

Blk mul(const Blk& A, const Blk& B)
{
  Blk C;
  C.r = A.r;
  C.c = B.c;
  for(int i = 0; i < A.r; i++)
    for(int j = 0; j < B.c; j++)
    {
      double s = 0;
      for(int k = 0; k < A.c; k++)
        s += A.a[i][k] * B.a[k][j];
      C.a[i][j] = s;
    }
  return C;
}

// A^T B
Blk tmul(const Blk& A, const Blk& B)
{
  Blk C;
  C.r = A.c;
  C.c = B.c;
  for(int i = 0; i < A.c; i++)
    for(int j = 0; j < B.c; j++)
    {
      double s = 0;
      for(int k = 0; k < A.r; k++)
        s += A.a[k][i] * B.a[k][j];
      C.a[i][j] = s;
    }
  return C;
}

// Gauss-Jordan with partial pivoting; a singular block gives what it can
Blk invert(Blk A)
{
  const int n = A.r;
  Blk       I;
  I.r = I.c = n;
  for(int i = 0; i < n; i++)
    I.a[i][i] = 1;
  for(int c = 0; c < n; c++)
  {
    int p = c;
    for(int r = c + 1; r < n; r++)
      if(std::abs(A.a[r][c]) > std::abs(A.a[p][c]))
        p = r;
    if(std::abs(A.a[p][c]) < 1e-300)
      continue;
    if(p != c)
      for(int k = 0; k < n; k++)
      {
        std::swap(A.a[p][k], A.a[c][k]);
        std::swap(I.a[p][k], I.a[c][k]);
      }
    const double d = 1 / A.a[c][c];
    for(int k = 0; k < n; k++)
    {
      A.a[c][k] *= d;
      I.a[c][k] *= d;
    }
    for(int r = 0; r < n; r++)
      if(r != c && A.a[r][c] != 0)
      {
        const double f = A.a[r][c];
        for(int k = 0; k < n; k++)
        {
          A.a[r][k] -= f * A.a[c][k];
          I.a[r][k] -= f * I.a[c][k];
        }
      }
  }
  return I;
}

// [I3, -[r]x] into rows `row` of J (the motion of a point r from a body's centre), times `sign`
void pointRows(Blk& J, int row, const Vec3& r, double sign)
{
  const double m[3][6] = {{1, 0, 0, 0, r.z, -r.y}, {0, 1, 0, -r.z, 0, r.x}, {0, 0, 1, r.y, -r.x, 0}};
  for(int i = 0; i < 3; i++)
    for(int k = 0; k < 6; k++)
      J.a[row + i][k] = sign * m[i][k];
}

// [0, I3] into rows `row` of J (a body's turn), times `sign`
void turnRows(Blk& J, int row, double sign)
{
  for(int i = 0; i < 3; i++)
    for(int k = 0; k < 6; k++)
      J.a[row + i][k] = k == 3 + i ? sign : 0;
}
}  // namespace

// The closest points of segments p1-q1 and p2-q2 (Ericson, Real-Time Collision Detection 5.1.9).
void closestSegments(const Vec3& p1, const Vec3& q1, const Vec3& p2, const Vec3& q2, Vec3& c1, Vec3& c2)
{
  const Vec3   d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
  const double a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
  double       s = 0, t = 0;
  if(a <= 1e-12 && e <= 1e-12)
    ;
  else if(a <= 1e-12)
    t = std::clamp(f / e, 0.0, 1.0);
  else
  {
    const double c = dot(d1, r);
    if(e <= 1e-12)
      s = std::clamp(-c / a, 0.0, 1.0);
    else
    {
      const double b = dot(d1, d2), denom = a * e - b * b;
      s              = denom > 1e-12 ? std::clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
      t              = (b * s + f) / e;
      if(t < 0)
      {
        t = 0;
        s = std::clamp(-c / a, 0.0, 1.0);
      }
      else if(t > 1)
      {
        t = 1;
        s = std::clamp((b - c) / a, 0.0, 1.0);
      }
    }
  }
  c1 = p1 + d1 * s;
  c2 = p2 + d2 * t;
}

Vec3 RigidBody::applyInvInertia(const Vec3& v) const
{
  const Quat f = q * inertiaFrame;
  return rotate(rotate(v, f.inverse()) * invInertia, f);
}

std::array<double, 3> eulerNear(const Quat& q, std::string_view order, const std::array<double, 3>& near)
{
  const std::array<double, 3> e   = eulerFromQuat(q.normalized(), order);
  const int                   mid = order.size() == 3 ? order[1] - 'X' : 1;
  std::array<double, 3>       best{};
  double                      bestCost = INFINITY;
  for(int other = 0; other < 2; other++)
  {
    std::array<double, 3> out{};
    double                cost = 0;
    for(int i = 0; i < 3; i++)
    {
      out[i] = other ? (i == mid ? kPi - e[i] : e[i] + kPi) : e[i];
      out[i] += 2 * kPi * std::round((near[i] - out[i]) / (2 * kPi));
      cost += (out[i] - near[i]) * (out[i] - near[i]);
    }
    if(cost < bestCost)
    {
      bestCost = cost;
      best     = out;
    }
  }
  return best;
}

void setCapsuleMass(RigidBody& body, double rho)
{
  const Capsule& c   = body.shape;
  const Vec3     ax  = c.b - c.a;
  const double   l   = length(ax), r = std::max(c.radius, 1e-3);
  const double   mc  = rho * kPi * r * r * l;
  const double   ms  = rho * 4.0 / 3.0 * kPi * r * r * r;
  const double   iz  = mc * r * r / 2 + ms * 2 * r * r / 5;
  const double   ixy = mc * (r * r / 4 + l * l / 12) + ms * (2 * r * r / 5 + l * l / 4 + 3 * l * r / 8);
  body.invMass       = 1 / (mc + ms);
  body.invInertia    = {1 / ixy, 1 / ixy, 1 / iz};
  body.inertiaFrame  = l > 1e-9 ? quatFromUnitVectors({0, 0, 1}, ax * (1 / l)) : Quat{};
}

void RigidWorld::wake()
{
  m_asleep = false;
  m_still  = 0;
}

void RigidWorld::sleep()
{
  m_asleep = true;
  for(RigidBody& b : bodies)
    b.v = b.w = {};
}

int RigidWorld::step(double seconds)
{
  if(m_asleep)
    return 0;
  m_h = settings.h;
  m_carry += std::clamp(seconds, 0.0, settings.maxFrame);
  const int n = int(std::floor(m_carry / m_h + 1e-9));
  m_carry     = std::max(0.0, m_carry - n * m_h);
  m_lastSpeed = 0;
  for(int i = 0; i < n && !m_asleep; i++)
    substep();
  return n;
}

void RigidWorld::substep()
{
  const double h = m_h = settings.h;
  predict(h);
  collide();
  // what holds the bodies together, solved exactly over their tree: with gravity's torque fed
  // forward, that alone puts a held pose back where it was, where sweeps of the constraints one by
  // one would leave a sag that shrinks only slowly with more of them. What closes a loop, and a pin
  // with a force limit (the pull), is swept after it.
  m_lambdaJoint.assign(joints.size() * 6, 0);
  m_lambdaPin.assign(pins.size() * 6, 0);
  std::vector<uint8_t> sweptJoint, sweptPin;
  for(int it = 0; it < std::max(1, settings.iterations); it++)
  {
    solveTree(sweptJoint, sweptPin);
    for(size_t i = 0; i < pins.size(); i++)
      if(pins[i].on && sweptPin[i])
        solvePin(pins[i], h);
    for(size_t i = 0; i < joints.size(); i++)
      if(sweptJoint[i])
      {
        solveAnchor(joints[i]);
        solveDrive(joints[i], h);
      }
  }
  for(Joint& j : joints)
    solveLimits(j);
  for(Contact& c : m_contacts)
    solveContact(c);
  updateVelocities(h);
  velocityPass(h);

  // asleep once nothing has moved faster than the rest speed for a while
  double speed = 0, spin = 0;
  for(const RigidBody& b : bodies)
  {
    if(b.invMass <= 0)
      continue;
    for(const Vec3& end : {b.shape.a, b.shape.b})
      speed = std::max(speed, length(b.v + cross(b.w, rotate(end, b.q))));
    spin = std::max(spin, length(b.w));
  }
  m_lastSpeed = std::max(m_lastSpeed, speed);
  m_still     = speed < settings.restSpeed && spin < settings.restSpin ? m_still + h : 0;
  if(m_still >= settings.restTime)
  {
    m_asleep = true;
    for(RigidBody& b : bodies)
      b.v = b.w = {};
  }
}

void RigidWorld::predict(double h)
{
  // (what joints and pins feed forward goes through their springs instead, in their solve)
  std::vector<Vec3> torque(bodies.size()), force(bodies.size());
  for(size_t i = 0; i < bodies.size(); i++)
  {
    torque[i] = bodies[i].torque;
    force[i]  = bodies[i].force;
  }
  for(size_t i = 0; i < bodies.size(); i++)
  {
    RigidBody& b = bodies[i];
    b.xPrev      = b.x;
    b.qPrev      = b.q;
    if(b.invMass <= 0)
      continue;
    b.v = b.v + (settings.gravity + force[i] * b.invMass) * h;
    b.w = b.w + b.applyInvInertia(torque[i]) * h;
    if(const double s = length(b.v); s > settings.maxSpeed)
      b.v = b.v * (settings.maxSpeed / s);
    if(const double s = length(b.w); s > settings.maxSpin)
      b.w = b.w * (settings.maxSpin / s);
    b.x = b.x + b.v * h;
    b.q = turned(b.q, b.w * h);
  }
}

double RigidWorld::applyPositional(int a, int b, const Vec3& pa, const Vec3& pb, const Vec3& n, double c, double alpha, double& lambda, double maxLambda)
{
  RigidBody* A = a >= 0 && bodies[size_t(a)].invMass > 0 ? &bodies[size_t(a)] : nullptr;
  RigidBody* B = b >= 0 && bodies[size_t(b)].invMass > 0 ? &bodies[size_t(b)] : nullptr;
  double     w = 0;
  if(A)
  {
    const Vec3 rn = cross(pa - A->x, n);
    w += A->invMass + dot(rn, A->applyInvInertia(rn));
  }
  if(B)
  {
    const Vec3 rn = cross(pb - B->x, n);
    w += B->invMass + dot(rn, B->applyInvInertia(rn));
  }
  const double alphaT = alpha / (m_h * m_h);
  if(w + alphaT <= 0)
    return 0;
  double dl = (c - alphaT * lambda) / (w + alphaT);
  if(maxLambda > 0)
    dl = std::clamp(lambda + dl, -maxLambda, maxLambda) - lambda;
  lambda += dl;
  applyPositionalImpulse(a, b, pa, pb, n * dl);
  return dl;
}

void RigidWorld::applyPositionalImpulse(int a, int b, const Vec3& pa, const Vec3& pb, const Vec3& p)
{
  // b's point moves along p, a's against it
  if(b >= 0 && bodies[size_t(b)].invMass > 0)
  {
    RigidBody& B = bodies[size_t(b)];
    const Vec3 r = pb - B.x;
    B.x          = B.x + p * B.invMass;
    B.q          = turned(B.q, B.applyInvInertia(cross(r, p)));
  }
  if(a >= 0 && bodies[size_t(a)].invMass > 0)
  {
    RigidBody& A = bodies[size_t(a)];
    const Vec3 r = pa - A.x;
    A.x          = A.x - p * A.invMass;
    A.q          = turned(A.q, A.applyInvInertia(cross(r, p)) * -1.0);
  }
}

double RigidWorld::applyAngular(int a, int b, const Vec3& n, double theta, double alpha, double& lambda, double maxLambda)
{
  RigidBody* A = a >= 0 && bodies[size_t(a)].invMass > 0 ? &bodies[size_t(a)] : nullptr;
  RigidBody* B = b >= 0 && bodies[size_t(b)].invMass > 0 ? &bodies[size_t(b)] : nullptr;
  double     w = 0;
  if(A)
    w += dot(n, A->applyInvInertia(n));
  if(B)
    w += dot(n, B->applyInvInertia(n));
  const double alphaT = alpha / (m_h * m_h);
  if(w + alphaT <= 0)
    return 0;
  double dl = (theta - alphaT * lambda) / (w + alphaT);
  if(maxLambda > 0)
    dl = std::clamp(lambda + dl, -maxLambda, maxLambda) - lambda;
  lambda += dl;
  const Vec3 p = n * dl;
  if(B)
    B->q = turned(B->q, B->applyInvInertia(p));
  if(A)
    A->q = turned(A->q, A->applyInvInertia(p) * -1.0);
  return dl;
}

void RigidWorld::solvePin(Pin& p, double h)
{
  if(p.body < 0 || bodies[size_t(p.body)].invMass <= 0)
    return;
  RigidBody& b = bodies[size_t(p.body)];
  if(p.move)
  {
    const Vec3   at = b.toWorld(p.local);
    const Vec3   d  = p.target + p.feedForward * p.compliance - at;
    const double c  = length(d);
    double       lambda = 0;
    if(c > 1e-12)
      applyPositional(-1, p.body, p.target, at, d * (1 / c), c, p.compliance, lambda, p.maxForce * h * h);
  }
  if(p.turn)
  {
    const Vec3   phi = rotationVector(p.rotation * b.q.inverse()) + p.feedForwardTorque * p.angularCompliance;
    const double th  = length(phi);
    double       lambda = 0;
    if(th > 1e-12)
      applyAngular(-1, p.body, phi * (1 / th), th, p.angularCompliance, lambda, p.maxTorque * h * h);
  }
}

void RigidWorld::solveTree(std::vector<uint8_t>& sweptJoint, std::vector<uint8_t>& sweptPin)
{
  sweptJoint.assign(joints.size(), 1);
  sweptPin.assign(pins.size(), 1);
  const double h2      = m_h * m_h;
  const auto   dynamic = [&](int b) { return b >= 0 && size_t(b) < bodies.size() && bodies[size_t(b)].invMass > 0; };
  // the constraints it can take: joints, and pins with no force limit; each with its dynamic bodies
  struct Con
  {
    bool joint = true;
    int  index = -1;
    int  rows  = 0;
    int  a = -1, b = -1;  // its dynamic bodies; a may be -1
  };
  std::vector<Con> cons;
  for(size_t i = 0; i < joints.size(); i++)
  {
    const Joint& j = joints[i];
    Con          c{true, int(i), j.compliance >= 0 ? 6 : 3, dynamic(j.a) ? j.a : -1, dynamic(j.b) ? j.b : -1};
    if(c.b < 0)
      std::swap(c.a, c.b);
    if(c.b >= 0)
      cons.push_back(c);
  }
  for(size_t i = 0; i < pins.size(); i++)
  {
    const Pin& p = pins[i];
    if(!p.on || p.maxForce > 0 || p.maxTorque > 0 || !dynamic(p.body) || (!p.move && !p.turn))
      continue;
    cons.push_back({false, int(i), (p.move ? 3 : 0) + (p.turn ? 3 : 0), -1, p.body});
  }
  if(cons.empty())
    return;
  // nodes: each body, then each constraint
  const int nb = int(bodies.size());
  struct Node
  {
    int              rows = 0, parent = -1;
    std::vector<int> children;
    // H_ii, H_i,parent, the factor's D and D^-1, and J_i = D_i^-1 H_i,parent
    Blk    H, Hp, D, Dinv, J;
    double b[6]{}, x[6]{};
    bool   in = false;
  };
  std::vector<Node>             nodes(size_t(nb) + cons.size());
  std::vector<std::vector<int>> conOf(static_cast<size_t>(nb));
  for(size_t c = 0; c < cons.size(); c++)
  {
    conOf[size_t(cons[c].b)].push_back(int(c));
    if(cons[c].a >= 0)
      conOf[size_t(cons[c].a)].push_back(int(c));
  }
  // the tree: from a one-body constraint (the hip's pin) where a group has one, so every leaf is a
  // body; a constraint that would close a loop, or a second one-body one, is left to the sweep
  std::vector<int>     order;
  std::vector<uint8_t> used(cons.size(), 0);
  const auto           grow = [&](int start) {
    std::vector<int> queue{start};
    nodes[size_t(start)].in = true;
    for(size_t q = 0; q < queue.size(); q++)
    {
      const int n = queue[q];
      order.push_back(n);
      if(n >= nb)
      {
        const Con& c = cons[size_t(n - nb)];
        for(const int body : {c.a, c.b})
          if(body >= 0 && !nodes[size_t(body)].in)
          {
            nodes[size_t(body)].in     = true;
            nodes[size_t(body)].parent = n;
            nodes[size_t(n)].children.push_back(body);
            queue.push_back(body);
          }
        continue;
      }
      for(const int ci : conOf[size_t(n)])
      {
        if(used[size_t(ci)])
          continue;
        const Con& c     = cons[size_t(ci)];
        const int  other = c.a == n ? c.b : c.a;
        used[size_t(ci)] = 1;
        if(other < 0 || nodes[size_t(other)].in)
          continue;  // a second one-body constraint, or a loop: swept
        const int cn             = nb + ci;
        nodes[size_t(cn)].in     = true;
        nodes[size_t(cn)].parent = n;
        nodes[size_t(n)].children.push_back(cn);
        queue.push_back(cn);
      }
    }
  };
  for(size_t c = 0; c < cons.size(); c++)
    if(cons[c].a < 0 && !nodes[size_t(cons[c].b)].in && !used[c])
    {
      used[c] = 1;
      grow(nb + int(c));
    }
  for(size_t c = 0; c < cons.size(); c++)
    if(!nodes[size_t(cons[c].b)].in)
      grow(cons[c].b);
  for(size_t c = 0; c < cons.size(); c++)
    if(nodes[size_t(nb) + c].in)
      (cons[c].joint ? sweptJoint : sweptPin)[size_t(cons[c].index)] = 0;

  // the blocks: a body's mass, a constraint's compliance (negated), and its value now
  for(const int n : order)
  {
    Node& node = nodes[size_t(n)];
    if(n < nb)
    {
      const RigidBody& b = bodies[size_t(n)];
      node.rows          = 6;
      node.H.r = node.H.c = 6;
      for(int i = 0; i < 3; i++)
        node.H.a[i][i] = 1 / b.invMass;
      // the inertia in the world: R diag(I) R^T
      const Quat   f          = b.q * b.inertiaFrame;
      const Vec3   axes[3]    = {rotate(Vec3{1, 0, 0}, f), rotate(Vec3{0, 1, 0}, f), rotate(Vec3{0, 0, 1}, f)};
      const double moments[3] = {1 / b.invInertia.x, 1 / b.invInertia.y, 1 / b.invInertia.z};
      for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
        {
          double s = 0;
          for(int k = 0; k < 3; k++)
          {
            const double ui = i == 0 ? axes[k].x : i == 1 ? axes[k].y : axes[k].z;
            const double uj = j == 0 ? axes[k].x : j == 1 ? axes[k].y : axes[k].z;
            s += ui * moments[k] * uj;
          }
          node.H.a[3 + i][3 + j] = s;
        }
      continue;
    }
    const Con& c = cons[size_t(n - nb)];
    node.rows    = c.rows;
    node.H.r = node.H.c = c.rows;
    double  C[6]{}, alpha[6]{}, gamma[6]{}, moved[6]{};
    double* lambda = c.joint ? &m_lambdaJoint[size_t(c.index) * 6] : &m_lambdaPin[size_t(c.index) * 6];
    if(c.joint)
    {
      const Joint&     j  = joints[size_t(c.index)];
      const RigidBody* A  = j.a >= 0 ? &bodies[size_t(j.a)] : nullptr;
      const RigidBody& B  = bodies[size_t(j.b)];
      const Vec3       pa = A ? A->toWorld(j.anchorA) : j.anchorA;
      const Vec3       pb = B.toWorld(j.anchorB);
      const Vec3       d  = pa - pb;
      C[0] = d.x, C[1] = d.y, C[2] = d.z;
      if(c.rows == 6)
      {
        const auto error = [&](const Quat& qa, const Quat& qb) {
          // a torque fed forward is the spring's rest moved by compliance times it: the same force
          // where it holds, and nothing large in the prediction to undo
          return rotationVector(qb * (qa * j.frameA * j.target * j.frameB.inverse()).inverse()) - j.feedForward * j.compliance;
        };
        const Vec3 e = error(A ? A->q : Quat{}, B.q);
        C[3] = e.x, C[4] = e.y, C[5] = e.z;
        alpha[3] = alpha[4] = alpha[5] = j.compliance / h2;
        // damped in the solve (Macklin et al. 2016: gamma = alpha c / h, and alpha c is the damping
        // time), so it reaches the whole chain the drive turns: damping a body's own spin in the
        // velocity pass reaches only it, a few percent of an arm's swing
        if(j.driveDamping > 0 && j.compliance > 0)
        {
          const Vec3 was = error(A ? A->qPrev : Quat{}, B.qPrev);
          moved[3] = e.x - was.x, moved[4] = e.y - was.y, moved[5] = e.z - was.z;
          gamma[3] = gamma[4] = gamma[5] = j.driveDamping / m_h;
        }
      }
    }
    else
    {
      const Pin&       p   = pins[size_t(c.index)];
      const RigidBody& b   = bodies[size_t(p.body)];
      int              row = 0;
      if(p.move)
      {
        const Vec3 d = b.toWorld(p.local) - p.target - p.feedForward * p.compliance;
        C[0] = d.x, C[1] = d.y, C[2] = d.z;
        alpha[0] = alpha[1] = alpha[2] = p.compliance / h2;
        row = 3;
      }
      if(p.turn)
      {
        const Vec3 e = rotationVector(b.q * p.rotation.inverse()) - p.feedForwardTorque * p.angularCompliance;
        C[row] = e.x, C[row + 1] = e.y, C[row + 2] = e.z;
        alpha[row] = alpha[row + 1] = alpha[row + 2] = p.angularCompliance / h2;
      }
    }
    // a damped row, divided through by 1 + gamma so the system stays symmetric
    for(int r = 0; r < c.rows; r++)
    {
      node.H.a[r][r] = -alpha[r] / (1 + gamma[r]);
      node.b[r]      = (C[r] + alpha[r] * lambda[r] + gamma[r] * moved[r]) / (1 + gamma[r]);
    }
  }
  // the Jacobian of constraint node `cn` for body `body`: how its value moves with the body's
  // move and turn
  const auto jacobian = [&](int cn, int body) {
    const Con& c = cons[size_t(cn - nb)];
    Blk        J;
    J.r                = c.rows;
    J.c                = 6;
    const RigidBody& B = bodies[size_t(body)];
    if(c.joint)
    {
      const Joint& j   = joints[size_t(c.index)];
      const bool   isA = j.a == body;
      pointRows(J, 0, rotate(isA ? j.anchorA : j.anchorB, B.q), isA ? 1 : -1);
      if(c.rows == 6)
        turnRows(J, 3, isA ? -1 : 1);
    }
    else
    {
      const Pin& p   = pins[size_t(c.index)];
      int        row = 0;
      if(p.move)
      {
        pointRows(J, 0, rotate(p.local, B.q), 1);
        row = 3;
      }
      if(p.turn)
        turnRows(J, row, 1);
    }
    return J;
  };
  // H_i,parent: -J^T for a body under a constraint, -J for a constraint under a body
  for(const int n : order)
  {
    Node& node = nodes[size_t(n)];
    if(node.parent < 0)
      continue;
    const Blk J = n < nb ? jacobian(node.parent, n) : jacobian(n, node.parent);
    if(n < nb)
    {
      node.Hp.r = J.c;
      node.Hp.c = J.r;
      for(int i = 0; i < J.r; i++)
        for(int k = 0; k < J.c; k++)
          node.Hp.a[k][i] = -J.a[i][k];
    }
    else
    {
      node.Hp = J;
      for(int i = 0; i < J.r; i++)
        for(int k = 0; k < J.c; k++)
          node.Hp.a[i][k] = -J.a[i][k];
    }
  }
  // factor, children before parents: D_i = H_ii - sum over its children j of H_ji^T J_j
  for(size_t k = order.size(); k-- > 0;)
  {
    Node& node = nodes[size_t(order[k])];
    node.D     = node.H;
    for(const int ch : node.children)
    {
      const Node& c   = nodes[size_t(ch)];
      const Blk   sub = tmul(c.Hp, c.J);
      for(int i = 0; i < node.D.r; i++)
        for(int j = 0; j < node.D.c; j++)
          node.D.a[i][j] -= sub.a[i][j];
    }
    node.Dinv = invert(node.D);
    if(node.parent >= 0)
      node.J = mul(node.Dinv, node.Hp);
  }
  // solve: forward (children first), the diagonal, then back (parents first)
  for(size_t k = order.size(); k-- > 0;)
  {
    Node& node = nodes[size_t(order[k])];
    for(int i = 0; i < node.rows; i++)
      node.x[i] = node.b[i];
    for(const int ch : node.children)
    {
      const Node& c = nodes[size_t(ch)];
      for(int i = 0; i < node.rows; i++)
        for(int r = 0; r < c.rows; r++)
          node.x[i] -= c.J.a[r][i] * c.x[r];
    }
  }
  for(const int n : order)
  {
    Node&  node = nodes[size_t(n)];
    double y[6];
    for(int i = 0; i < node.rows; i++)
    {
      y[i] = 0;
      for(int k = 0; k < node.rows; k++)
        y[i] += node.Dinv.a[i][k] * node.x[k];
    }
    for(int i = 0; i < node.rows; i++)
      node.x[i] = y[i];
  }
  for(const int n : order)
  {
    Node& node = nodes[size_t(n)];
    if(node.parent < 0)
      continue;
    const Node& p = nodes[size_t(node.parent)];
    for(int i = 0; i < node.rows; i++)
      for(int k = 0; k < p.rows; k++)
        node.x[i] -= node.J.a[i][k] * p.x[k];
  }
  // the bodies move and turn, the multipliers add up
  for(const int n : order)
  {
    const Node& node = nodes[size_t(n)];
    if(n < nb)
    {
      RigidBody& b = bodies[size_t(n)];
      b.x          = b.x + Vec3{node.x[0], node.x[1], node.x[2]};
      b.q          = turned(b.q, Vec3{node.x[3], node.x[4], node.x[5]});
      continue;
    }
    const Con& c      = cons[size_t(n - nb)];
    double*    lambda = c.joint ? &m_lambdaJoint[size_t(c.index) * 6] : &m_lambdaPin[size_t(c.index) * 6];
    for(int r = 0; r < c.rows; r++)
      lambda[r] += node.x[r];
  }
}

void RigidWorld::solveAnchor(Joint& j)
{
  const RigidBody* A  = j.a >= 0 ? &bodies[size_t(j.a)] : nullptr;
  const Vec3       pa = A ? A->toWorld(j.anchorA) : j.anchorA;
  const Vec3       pb = bodies[size_t(j.b)].toWorld(j.anchorB);
  const Vec3       d  = pa - pb;
  const double     c  = length(d);
  double           lambda = 0;
  if(c > 1e-12)
    applyPositional(j.a, j.b, pa, pb, d * (1 / c), c, 0, lambda, 0);
}

void RigidWorld::solveDrive(Joint& j, double h)
{
  if(j.compliance < 0)
    return;
  const RigidBody* A      = j.a >= 0 ? &bodies[size_t(j.a)] : nullptr;
  const Quat       want   = (A ? A->q : Quat{}) * j.frameA * j.target * j.frameB.inverse();
  const Vec3       phi    = rotationVector(want * bodies[size_t(j.b)].q.inverse()) + j.feedForward * j.compliance;
  const double     th     = length(phi);
  double           lambda = 0;
  if(th > 1e-12)
    applyAngular(j.a, j.b, phi * (1 / th), th, j.compliance, lambda, j.maxTorque * h * h);
}

void RigidWorld::solveLimits(Joint& j)
{
  const RigidBody* A   = j.a >= 0 ? &bodies[size_t(j.a)] : nullptr;
  const auto       rel = [&](const Quat& qb) { return j.frameA.inverse() * (A ? A->q : Quat{}).inverse() * qb * j.frameB; };
  j.euler              = eulerNear(rel(bodies[size_t(j.b)].q), j.order, j.euler);
  for(int k = 0; k < 3; k++)
  {
    if(j.lo[k] > j.hi[k])
      continue;
    // where the joint is now: an earlier axis's correction turned it
    const auto e = eulerNear(rel(bodies[size_t(j.b)].q), j.order, j.euler);
    j.euler      = e;
    const double bound = e[k] < j.lo[k] ? j.lo[k] : e[k] > j.hi[k] ? j.hi[k] : e[k];
    if(bound == e[k])
      continue;
    // the gradient of this Euler angle for a turn of b in the world: a row of the inverse Euler-rate
    // map, numerically. A correction along it doesn't slide the joint along its other axes.
    constexpr double eps = 1e-6;
    Vec3             g;
    for(int i = 0; i < 3; i++)
    {
      const Vec3 axis{double(i == 0), double(i == 1), double(i == 2)};
      const auto turnedE = eulerNear(rel(fromRotationVector(axis * eps) * bodies[size_t(j.b)].q), j.order, e);
      (i == 0 ? g.x : i == 1 ? g.y : g.z) = wrapPi(turnedE[size_t(k)] - e[size_t(k)]) / eps;
    }
    const double g2 = dot(g, g);
    if(g2 < 1e-18)
      continue;
    // damped near the gimbal, where the outer axes turn fast for a small turn of the body
    const double gl  = std::sqrt(g2);
    const double rot = (bound - e[size_t(k)]) * gl / (g2 + 0.01);
    double       lambda = 0;
    applyAngular(j.a, j.b, g * (1 / gl), rot, 0, lambda, 0);
  }
}

void RigidWorld::collide()
{
  m_contacts.clear();
  const auto allowed = [&](int a, int b) -> double {
    for(const Allowance& al : allowances)
      if(al.a == a && al.b == b)
        return al.depth;
    return 0;
  };
  const auto shrink = [&](int a, int b, double depth) {
    for(Allowance& al : allowances)
      if(al.a == a && al.b == b)
        al.depth = std::min(al.depth, std::max(0.0, depth));
  };
  std::vector<double> fixedDepth(bodies.size(), 0);
  for(size_t i = 0; i < bodies.size(); i++)
  {
    const RigidBody& b = bodies[i];
    if(b.invMass <= 0 || b.shape.radius <= 0)
      continue;
    const double r  = b.shape.radius;
    const Vec3   ea = b.toWorld(b.shape.a), eb = b.toWorld(b.shape.b);
    const auto   add = [&](int other, const Vec3& onA, const Vec3& onB, const Vec3& n, double depth) {
      Contact c;
      c.a     = int(i);
      c.b     = other;
      c.ra    = rotate(onA - b.x, b.q.inverse());
      c.rb    = other >= 0 ? rotate(onB - bodies[size_t(other)].x, bodies[size_t(other)].q.inverse()) : onB;
      c.n     = n;
      c.depth = depth;
      if(other < 0)
        fixedDepth[i] = std::max(fixedDepth[i], depth);
      m_contacts.push_back(c);
    };
    for(const Plane& pl : planes)
      for(const Vec3& e : {ea, eb})
        if(const double d = dot(pl.n, e) - pl.offset - r; d < 0)
          add(-1, e - pl.n * r, e - pl.n * (r + d), pl.n, -d);
    // props: the axis sampled at about a radius apart, each sample a sphere
    for(const ColliderRef& prop : props)
    {
      if(!prop.set)
        continue;
      Box capsuleBox;
      capsuleBox.add(ea.x, ea.y, ea.z);
      capsuleBox.add(eb.x, eb.y, eb.z);
      if(!transformed(prop.set->bounds(), prop.toBody).overlaps(capsuleBox.grown(r)))
        continue;
      const TriangleSet& set     = prop.set->get();
      const Mat4         toProp  = inverse(prop.toBody);
      const double       scale   = length(linearPart(prop.toBody, {1, 0, 0}));
      const double       rp      = r / std::max(scale, 1e-9);
      const int          samples = std::max(2, int(std::ceil(length(eb - ea) / r)) + 1);
      for(int s = 0; s < samples; s++)
      {
        const Vec3   w = ea + (eb - ea) * (double(s) / (samples - 1));
        const Vec3   p = toProp.transformPoint(w);
        const Inside in = insideOf(set, {p.x, p.y, p.z}, rp, 8 / std::max(scale, 1e-9), threadStamps());
        if(in.inside)
        {
          const Vec3 n = normalize(linearPart(prop.toBody, {in.normal[0], in.normal[1], in.normal[2]}));
          add(-1, w - n * r, prop.toBody.transformPoint({in.exit[0], in.exit[1], in.exit[2]}), n, in.depth * scale + r);
          continue;
        }
        Stamps&                  st  = threadStamps();
        const int32_t            id  = st.next(set.index().triangles());
        const SurfaceIndex::Hit hit = set.index().search(p.x, p.y, p.z, 0, {}, rp * rp, st.stamp, id);
        if(hit.tri < 0 || hit.d2 >= rp * rp)
          continue;
        const auto     P   = set.positions();
        const uint32_t* t  = &set.tris()[size_t(hit.tri) * 3];
        const auto     at  = [&](uint32_t v) { return Vec3{P[v * 3], P[v * 3 + 1], P[v * 3 + 2]}; };
        const Vec3     cp  = at(t[0]) * hit.u + at(t[1]) * hit.v + at(t[2]) * hit.w;
        const Vec3     cw  = prop.toBody.transformPoint(cp);
        const double   d   = length(w - cw);
        if(d < 1e-9)
          continue;
        const Vec3 n = (w - cw) * (1 / d);
        add(-1, w - n * r, cw, n, r - d);
      }
    }
    // the other bodies
    for(size_t j = i + 1; j < bodies.size(); j++)
    {
      const RigidBody& o = bodies[j];
      if(o.shape.radius <= 0 || std::find(b.noCollide.begin(), b.noCollide.end(), int(j)) != b.noCollide.end()
         || std::find(o.noCollide.begin(), o.noCollide.end(), int(i)) != o.noCollide.end())
        continue;
      Vec3 ci, cj;
      closestSegments(ea, eb, o.toWorld(o.shape.a), o.toWorld(o.shape.b), ci, cj);
      const double d   = length(ci - cj);
      const double pen = r + o.shape.radius - d;
      const double ok  = allowed(int(i), int(j));
      shrink(int(i), int(j), pen);
      if(pen <= ok || d < 1e-9)
        continue;
      const Vec3 n = (ci - cj) * (1 / d);
      Contact    c;
      c.a       = int(i);
      c.b       = int(j);
      c.ra      = rotate(ci - n * r - b.x, b.q.inverse());
      c.rb      = rotate(cj + n * o.shape.radius - o.x, o.q.inverse());
      c.n       = n;
      c.depth   = pen;
      c.allowed = std::min(ok, pen);
      m_contacts.push_back(c);
    }
  }
  // what a body is allowed into fixed things (it came in seated in a chair, its feet a hair under the
  // ground), shrinking as it leaves them
  for(Contact& c : m_contacts)
    if(c.b < 0)
      c.allowed = std::min(allowed(c.a, -1), c.depth);
  for(size_t i = 0; i < bodies.size(); i++)
    shrink(int(i), -1, fixedDepth[i]);
}

void RigidWorld::solveContact(Contact& c)
{
  RigidBody&   A  = bodies[size_t(c.a)];
  const Vec3   pa = A.toWorld(c.ra);
  const Vec3   pb = c.b >= 0 ? bodies[size_t(c.b)].toWorld(c.rb) : c.rb;
  const double gap = dot(pa - pb, c.n);
  const double corr = -gap - c.allowed;
  if(corr <= 0)
    return;
  // a's point moves along n, b's against it
  applyPositional(c.b, c.a, pb, pa, c.n, corr, 0, c.lambdaN, 0);
  // static friction: the points don't slide over the substep while the friction holds
  const Vec3 paNow  = A.toWorld(c.ra);
  const Vec3 paPrev = A.xPrev + rotate(c.ra, A.qPrev);
  Vec3       dp     = paNow - paPrev;
  if(c.b >= 0)
  {
    const RigidBody& B = bodies[size_t(c.b)];
    dp                 = dp - (B.toWorld(c.rb) - (B.xPrev + rotate(c.rb, B.qPrev)));
  }
  const Vec3   dt = dp - c.n * dot(dp, c.n);
  const double l  = length(dt);
  if(l < 1e-12)
    return;
  const Vec3 n = dt * (-1 / l);
  double     w = 0;
  {
    const Vec3 rn = cross(paNow - A.x, n);
    w += A.invMass + dot(rn, A.applyInvInertia(rn));
  }
  if(c.b >= 0 && bodies[size_t(c.b)].invMass > 0)
  {
    const RigidBody& B  = bodies[size_t(c.b)];
    const Vec3       rn = cross(B.toWorld(c.rb) - B.x, n);
    w += B.invMass + dot(rn, B.applyInvInertia(rn));
  }
  if(w > 0 && l / w < settings.material.staticFriction * c.lambdaN)
    applyPositional(c.b, c.a, c.b >= 0 ? bodies[size_t(c.b)].toWorld(c.rb) : pb, paNow, n, l, 0, c.lambdaT, 0);
}

void RigidWorld::updateVelocities(double h)
{
  for(RigidBody& b : bodies)
  {
    if(b.invMass <= 0)
      continue;
    b.v       = (b.x - b.xPrev) * (1 / h);
    Quat dq   = b.q * b.qPrev.inverse();
    if(dq.w < 0)
      dq = {-dq.x, -dq.y, -dq.z, -dq.w};
    b.w = Vec3{dq.x, dq.y, dq.z} * (2 / h);
  }
}

void RigidWorld::velocityPass(double h)
{
  const Material& m = settings.material;
  for(const Contact& c : m_contacts)
  {
    if(c.lambdaN <= 0)
      continue;
    RigidBody&   A  = bodies[size_t(c.a)];
    RigidBody*   B  = c.b >= 0 && bodies[size_t(c.b)].invMass > 0 ? &bodies[size_t(c.b)] : nullptr;
    const Vec3   ra = rotate(c.ra, A.q);
    const Vec3   rb = B ? rotate(c.rb, B->q) : Vec3{};
    const Vec3   vr = (A.v + cross(A.w, ra)) - (B ? B->v + cross(B->w, rb) : Vec3{});
    const double vn = dot(vr, c.n);
    const Vec3   vt = vr - c.n * vn;
    const double fn = c.lambdaN / (h * h);
    // no bounce, and dynamic friction up to the normal force
    Vec3 dv;
    if(vn < 0)
      dv = c.n * -vn;
    if(const double lt = length(vt); lt > 1e-12)
      dv = dv - vt * (std::min(h * m.dynamicFriction * fn, lt) / lt);
    const double ldv = length(dv);
    if(ldv > 1e-12)
    {
      const Vec3 n  = dv * (1 / ldv);
      const Vec3 an = cross(ra, n);
      double     w  = A.invMass + dot(an, A.applyInvInertia(an));
      if(B)
      {
        const Vec3 bn = cross(rb, n);
        w += B->invMass + dot(bn, B->applyInvInertia(bn));
      }
      const Vec3 p = dv * (1 / w);
      A.v          = A.v + p * A.invMass;
      A.w          = A.w + A.applyInvInertia(cross(ra, p));
      if(B)
      {
        B->v = B->v - p * B->invMass;
        B->w = B->w - B->applyInvInertia(cross(rb, p));
      }
    }
    // rolling resistance against something fixed, so a slump comes to rest
    if(!B)
    {
      const Vec3   roll = A.w - c.n * dot(A.w, c.n);
      const double lr   = length(roll);
      if(lr > 1e-12)
      {
        const Vec3   u    = roll * (1 / lr);
        const double wAng = dot(u, A.applyInvInertia(u));
        const double cut  = std::min(lr, m.rolling * fn * A.shape.radius * h * wAng);
        A.w               = A.w - u * cut;
      }
    }
  }
  for(const Joint& j : joints)
  {
    if(j.damping <= 0)
      continue;
    RigidBody* A  = j.a >= 0 && bodies[size_t(j.a)].invMass > 0 ? &bodies[size_t(j.a)] : nullptr;
    RigidBody& B  = bodies[size_t(j.b)];
    if(B.invMass <= 0)
      continue;
    const Vec3   dw = (B.w - (A ? A->w : Vec3{})) * std::min(1.0, j.damping * h);
    const double l  = length(dw);
    if(l < 1e-12)
      continue;
    const Vec3 n = dw * (1 / l);
    double     w = dot(n, B.applyInvInertia(n)) + (A ? dot(n, A->applyInvInertia(n)) : 0);
    const Vec3 L = n * (l / w);
    B.w          = B.w - B.applyInvInertia(L);
    if(A)
      A->w = A->w + A->applyInvInertia(L);
  }
  for(const Pin& p : pins)
    if(p.on && p.damping > 0 && p.body >= 0 && bodies[size_t(p.body)].invMass > 0)
    {
      RigidBody&   b = bodies[size_t(p.body)];
      const double k = std::exp(-p.damping * h);
      b.v            = b.v * k;
      if(p.turn)
        b.w = b.w * k;
    }
}

double RigidWorld::overlap(int a, int b) const
{
  const RigidBody& A = bodies[size_t(a)];
  const RigidBody& B = bodies[size_t(b)];
  if(A.shape.radius <= 0 || B.shape.radius <= 0)
    return 0;
  Vec3 ca, cb;
  closestSegments(A.toWorld(A.shape.a), A.toWorld(A.shape.b), B.toWorld(B.shape.a), B.toWorld(B.shape.b), ca, cb);
  return std::max(0.0, A.shape.radius + B.shape.radius - length(ca - cb));
}

void RigidWorld::allowOverlaps()
{
  allowances.clear();
  for(size_t i = 0; i < bodies.size(); i++)
    for(size_t j = i + 1; j < bodies.size(); j++)
    {
      const RigidBody& b = bodies[i];
      if(std::find(b.noCollide.begin(), b.noCollide.end(), int(j)) != b.noCollide.end())
        continue;
      if(const double d = overlap(int(i), int(j)); d > 0)
        allowances.push_back({int(i), int(j), d});
    }
  // and into the ground and the props: the contacts as they stand now
  for(RigidBody& b : bodies)
    b.xPrev = b.x, b.qPrev = b.q;
  const std::vector<Allowance> pairs = allowances;
  allowances.clear();
  collide();
  std::vector<double> fixed(bodies.size(), 0);
  for(const Contact& c : m_contacts)
    if(c.b < 0)
      fixed[size_t(c.a)] = std::max(fixed[size_t(c.a)], c.depth);
  allowances = pairs;
  for(size_t i = 0; i < bodies.size(); i++)
    if(fixed[i] > 0)
      allowances.push_back({int(i), -1, fixed[i]});
  m_contacts.clear();
}

}  // namespace fv::physics
