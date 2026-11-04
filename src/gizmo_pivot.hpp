#pragma once

// A transform moved, turned and scaled by the gizmo about a point of its own (GizmoHooks::pivot)
// rather than its origin: the gizmo shows and drags that point, and the transform follows it.

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

// Where `pivot` (in the transform's own space) is under the transform t, r, s.
inline glm::vec3 pivotPosition(const glm::vec3& t, const glm::quat& r, const glm::vec3& s, const glm::vec3& pivot)
{
  return t + r * (s * pivot);
}

// The translation that puts `pivot` at `at` under the rotation r and scale s.
inline glm::vec3 translationFromPivot(const glm::vec3& at, const glm::quat& r, const glm::vec3& s, const glm::vec3& pivot)
{
  return at - r * (s * pivot);
}
