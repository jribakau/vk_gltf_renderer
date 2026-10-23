#include <figure/pose/formulas.hpp>

#include <algorithm>

namespace fv::pose {

namespace {
// knot[i] as the TS reads it: a missing x or y is undefined (NaN), a missing tension, continuity or
// bias is 0
double at(const std::vector<double>& k, size_t i)
{
  return i < k.size() ? k[i] : (i < 2 ? NAN : 0.0);
}
}  // namespace

double splineTcb(double x, std::vector<const std::vector<double>*> knots, bool linear)
{
  // DSON writes knots in order; only an unordered list pays for a sorted copy
  auto less = [](const std::vector<double>* a, const std::vector<double>* b) { return at(*a, 0) < at(*b, 0); };
  if(!std::is_sorted(knots.begin(), knots.end(), less))
    std::stable_sort(knots.begin(), knots.end(), less);
  if(knots.empty())
    return 0;
  auto K = [&](size_t i) -> const std::vector<double>& { return *knots[i]; };
  if(x <= at(K(0), 0))
    return at(K(0), 1);
  const auto& last = K(knots.size() - 1);
  if(x >= at(last, 0))
    return at(last, 1);
  size_t i = 0;
  while(i + 2 < knots.size() && x > at(K(i + 1), 0))
    i++;
  const auto&  p1 = K(i);
  const auto&  p2 = K(i + 1);
  const double d  = at(p2, 0) - at(p1, 0);
  const double u  = (x - at(p1, 0)) / (d != 0 ? d : 1);
  if(linear)
    return at(p1, 1) + (at(p2, 1) - at(p1, 1)) * u;
  const auto& p0      = i > 0 ? K(i - 1) : p1;
  const auto& p3      = i + 2 < knots.size() ? K(i + 2) : p2;
  auto        tangent = [](const std::vector<double>& prev, const std::vector<double>& cur, const std::vector<double>& next, bool out) {
    const double t = at(cur, 2), c = at(cur, 3), b = at(cur, 4);
    const double a = out ? ((1 - t) * (1 + b) * (1 - c)) / 2 : ((1 - t) * (1 + b) * (1 + c)) / 2;
    const double e = out ? ((1 - t) * (1 - b) * (1 + c)) / 2 : ((1 - t) * (1 - b) * (1 - c)) / 2;
    return a * (at(cur, 1) - at(prev, 1)) + e * (at(next, 1) - at(cur, 1));
  };
  const double m1 = tangent(p0, p1, p2, true);
  const double m2 = tangent(p1, p2, p3, false);
  const double u2 = u * u;
  const double u3 = u2 * u;
  return (2 * u3 - 3 * u2 + 1) * at(p1, 1) + (u3 - 2 * u2 + u) * m1 + (-2 * u3 + 3 * u2) * at(p2, 1) + (u3 - u2) * m2;
}

}  // namespace fv::pose
