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

Vec3 linearPart(const Mat4& m, const Vec3& v)
{
  return {m.e[0] * v.x + m.e[4] * v.y + m.e[8] * v.z, m.e[1] * v.x + m.e[5] * v.y + m.e[9] * v.z, m.e[2] * v.x + m.e[6] * v.y + m.e[10] * v.z};
}

}  // namespace

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
  for(Pin& p : pins)
    if(p.on)
      solvePin(p, h);
  for(Joint& j : joints)
    solveJoint(j, h);
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
  // what joints and pins add to the bodies' own forces
  std::vector<Vec3> torque(bodies.size()), force(bodies.size());
  for(size_t i = 0; i < bodies.size(); i++)
  {
    torque[i] = bodies[i].torque;
    force[i]  = bodies[i].force;
  }
  for(const Joint& j : joints)
  {
    if(j.b >= 0)
      torque[size_t(j.b)] = torque[size_t(j.b)] + j.feedForward;
    if(j.a >= 0)
      torque[size_t(j.a)] = torque[size_t(j.a)] - j.feedForward;
  }
  for(const Pin& p : pins)
    if(p.on && p.body >= 0)
    {
      force[size_t(p.body)]  = force[size_t(p.body)] + p.feedForward;
      torque[size_t(p.body)] = torque[size_t(p.body)] + p.feedForwardTorque;
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
    const Vec3   d  = p.target - at;
    const double c  = length(d);
    double       lambda = 0;
    if(c > 1e-12)
      applyPositional(-1, p.body, p.target, at, d * (1 / c), c, p.compliance, lambda, p.maxForce * h * h);
  }
  if(p.turn)
  {
    const Vec3   phi = rotationVector(p.rotation * b.q.inverse());
    const double th  = length(phi);
    double       lambda = 0;
    if(th > 1e-12)
      applyAngular(-1, p.body, phi * (1 / th), th, p.angularCompliance, lambda, p.maxTorque * h * h);
  }
}

void RigidWorld::solveJoint(Joint& j, double h)
{
  const RigidBody* A = j.a >= 0 ? &bodies[size_t(j.a)] : nullptr;
  const RigidBody& B = bodies[size_t(j.b)];
  const auto       anchor = [&] {
    const Vec3   pa = A ? A->toWorld(j.anchorA) : j.anchorA;
    const Vec3   pb = B.toWorld(j.anchorB);
    const Vec3   d  = pa - pb;
    const double c  = length(d);
    double       lambda = 0;
    if(c > 1e-12)
      applyPositional(j.a, j.b, pa, pb, d * (1 / c), c, 0, lambda, 0);
    return lambda;
  };
  // the anchor first: with gravity's torque fed forward, it alone puts a held limb back where it
  // was, so the drive then sees no error to pull at. A drive before it would leave a sag of its own.
  j.lambda = anchor();
  if(j.compliance >= 0)
  {
    const Quat   want   = (A ? A->q : Quat{}) * j.frameA * j.target * j.frameB.inverse();
    const Vec3   phi    = rotationVector(want * B.q.inverse());
    const double th     = length(phi);
    double       lambda = 0;
    if(th > 1e-12)
    {
      applyAngular(j.a, j.b, phi * (1 / th), th, j.compliance, lambda, j.maxTorque * h * h);
      // the drive turned the bodies about their centres: the anchor again, so it has the last word
      j.lambda += anchor();
    }
  }
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
