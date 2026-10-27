/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scene_clip_player.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include <glm/gtc/type_ptr.hpp>

namespace rs {

namespace {

// Where `time` falls among the keys: the segment from key `index` to the next, the factor in it
// and its length (AnimationSystem::processAnimationChannel and calculateInterpolationFactor).
// Nothing before the first key, after the last, or with fewer than two keys.
struct Segment
{
  size_t index    = 0;
  float  t        = 0.f;
  float  keyDelta = 0.f;
  size_t step     = 0;  // the key a STEP channel holds: `index`, or the last exactly on it
};

std::optional<Segment> segmentAt(const std::vector<float>& inputs, float time)
{
  if(inputs.size() < 2)
    return {};
  const auto it = std::upper_bound(inputs.begin(), inputs.end(), time);
  if(it == inputs.begin())
    return {};
  size_t i = size_t(std::distance(inputs.begin(), it)) - 1;
  if(i + 1 >= inputs.size())
    i = inputs.size() - 2;
  const float inputStart = inputs[i];
  const float inputEnd   = inputs[i + 1];
  if(time < inputStart || time > inputEnd)
    return {};
  const float keyDelta = inputEnd - inputStart;
  const float t = std::abs(keyDelta) < std::numeric_limits<float>::epsilon() ? 0.0f : std::clamp((time - inputStart) / keyDelta, 0.0f, 1.0f);
  return Segment{i, t, keyDelta, time >= inputEnd ? i + 1 : i};
}

// glTF's cubic Hermite spline between value v0 (out-tangent b) and value v1 (in-tangent a), the
// tangents scaled by the segment's length (computeCubicInterpolation)
template <class T>
T cubic(const T& v0, const T& a, const T& b, const T& v1, float t, float keyDelta)
{
  const float tSq = t * t;
  const float tCb = tSq * t;
  const float tD  = keyDelta;
  const float cV1 = -2 * tCb + 3 * tSq;
  const float cV0 = 1 - cV1;
  const float cA  = tD * (tCb - tSq);
  const float cB  = tD * (tCb - 2 * tSq + t);
  return v0 * cV0 + a * cA + b * cB + v1 * cV1;
}

glm::vec3 vec3At(const std::vector<float>& v, size_t k)
{
  return glm::make_vec3(v.data() + 3 * k);
}

glm::vec4 vec4At(const std::vector<float>& v, size_t k)
{
  return glm::make_vec4(v.data() + 4 * k);
}

// a translation or scale at the segment (handle*Interpolation's bounds: nothing when its keys are short)
std::optional<glm::vec3> evalVec3(const ClipSampler& s, const Segment& g)
{
  const size_t count = s.outputs.size() / 3;
  const size_t i     = g.index;
  switch(s.interpolation)
  {
    case ClipInterpolation::eLinear:
      if(i + 1 >= count)
        return {};
      return glm::mix(vec3At(s.outputs, i), vec3At(s.outputs, i + 1), g.t);
    case ClipInterpolation::eStep:
      if(g.step >= count)
        return {};
      return vec3At(s.outputs, g.step);
    case ClipInterpolation::eCubicSpline:
      if(count <= (i + 1) * 3 + 1)
        return {};
      return cubic(vec3At(s.outputs, i * 3 + 1), vec3At(s.outputs, (i + 1) * 3), vec3At(s.outputs, i * 3 + 2),
                   vec3At(s.outputs, (i + 1) * 3 + 1), g.t, g.keyDelta);
  }
  return {};
}

std::optional<glm::quat> evalRotation(const ClipSampler& s, const Segment& g)
{
  const size_t count = s.outputs.size() / 4;
  const size_t i     = g.index;
  switch(s.interpolation)
  {
    case ClipInterpolation::eLinear: {
      if(i + 1 >= count)
        return {};
      const glm::quat q1 = glm::make_quat(s.outputs.data() + 4 * i);
      const glm::quat q2 = glm::make_quat(s.outputs.data() + 4 * (i + 1));
      return glm::normalize(glm::slerp(q1, q2, g.t));
    }
    case ClipInterpolation::eStep:
      if(g.step >= count)
        return {};
      return glm::make_quat(s.outputs.data() + 4 * g.step);
    case ClipInterpolation::eCubicSpline: {
      if(count <= (i + 1) * 3 + 1)
        return {};
      const glm::vec4 r = cubic(vec4At(s.outputs, i * 3 + 1), vec4At(s.outputs, (i + 1) * 3), vec4At(s.outputs, i * 3 + 2),
                                vec4At(s.outputs, (i + 1) * 3 + 1), g.t, g.keyDelta);
      return glm::normalize(glm::make_quat(glm::value_ptr(r)));
    }
  }
  return {};
}

// Morph weights: one per target in each key (three per target for a cubic spline: the in-tangents,
// the values, then the out-tangents of every target)
std::optional<std::vector<float>> evalWeights(const ClipSampler& s, const Segment& g)
{
  const size_t perKey = s.outputs.size() / s.inputs.size();
  const size_t i      = g.index;
  const float* key0   = s.outputs.data() + i * perKey;
  const float* key1   = key0 + perKey;
  std::vector<float> out;
  switch(s.interpolation)
  {
    case ClipInterpolation::eLinear:
      for(size_t j = 0; j < perKey; j++)
        out.push_back(glm::mix(key0[j], key1[j], g.t));
      return out;
    case ClipInterpolation::eStep: {
      const float* key = s.outputs.data() + g.step * perKey;
      return std::vector<float>(key, key + perKey);
    }
    case ClipInterpolation::eCubicSpline: {
      if(perKey % 3 != 0)
        return {};
      const size_t count = perKey / 3;
      for(size_t j = 0; j < count; j++)
        out.push_back(cubic(key0[count + j], key1[j], key0[2 * count + j], key1[count + j], g.t, g.keyDelta));
      return out;
    }
  }
  return {};
}

}  // namespace

ClipPlayer::ClipPlayer(const RetainedScene& scene, ClipId clip)
    : m_clip(clip)
{
  if(const ClipRecord* c = scene.clip(clip))
  {
    const std::array<float, 2> range = c->range();
    if(range[0] <= range[1])
    {
      m_start = range[0];
      m_end   = range[1];
    }
  }
  m_time = m_start;
}

float ClipPlayer::advance(float deltaTime, bool loop)
{
  m_time += deltaTime;
  if(loop)
  {
    const float duration = m_end - m_start;
    if(!(duration > 0.f))
      return m_time = m_start;
    float wrapped = std::fmod(m_time - m_start, duration);
    // fmod is negative for a time before the start
    if(wrapped < 0.0f)
      wrapped += duration;
    m_time = m_start + wrapped;
  }
  else if(m_time > m_end)
    m_time = m_end;
  return m_time;
}

bool ClipPlayer::apply(RetainedScene& scene) const
{
  const ClipRecord* clip = scene.clip(m_clip);
  if(!clip)
    return false;
  bool wrote = false;
  for(const ClipChannel& channel : clip->channels)
  {
    const Entity* e = scene.entity(channel.target);
    if(!e || channel.sampler >= clip->samplers.size())
      continue;
    const ClipSampler&           sampler = clip->samplers[channel.sampler];
    const std::optional<Segment> segment = segmentAt(sampler.inputs, m_time);
    if(!segment)
      continue;
    Transform local = e->local;
    switch(channel.path)
    {
      case ClipPath::eTranslation:
      case ClipPath::eScale:
        if(const std::optional<glm::vec3> v = evalVec3(sampler, *segment))
        {
          (channel.path == ClipPath::eTranslation ? local.translation : local.scale) = *v;
          scene.setLocal(channel.target, local);
          wrote = true;
        }
        break;
      case ClipPath::eRotation:
        if(const std::optional<glm::quat> q = evalRotation(sampler, *segment))
        {
          local.rotation = *q;
          scene.setLocal(channel.target, local);
          wrote = true;
        }
        break;
      case ClipPath::eWeights:
        // the entity's own weights, over its meshes' defaults; an entity drawing nothing has none
        if(e->mesh)
          if(std::optional<std::vector<float>> w = evalWeights(sampler, *segment))
          {
            scene.setMorphWeights(channel.target, std::move(*w));
            wrote = true;
          }
        break;
    }
  }
  return wrote;
}

}  // namespace rs
