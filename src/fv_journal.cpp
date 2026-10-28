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

#include "fv_journal.hpp"

namespace fv {

void Journal::record(JournalStep step)
{
  m_redo.clear();
  const bool sealed = std::exchange(m_sealed, false);
  if(!sealed && !step.key.empty() && !m_undo.empty())
  {
    JournalStep& last = m_undo.back();
    if(last.key == step.key && step.at - last.at <= kMerge)
    {
      last.redo = std::move(step.redo);
      last.at   = step.at;
      // back where it began: nothing to undo
      if(last.redo == last.undo)
        m_undo.pop_back();
      return;
    }
  }
  m_undo.push_back(std::move(step));
  while(m_undo.size() > kLimit)
    m_undo.pop_front();
}

std::optional<JournalStep> Journal::takeUndo()
{
  if(m_undo.empty())
    return std::nullopt;
  m_redo.push_back(std::move(m_undo.back()));
  m_undo.pop_back();
  m_sealed = true;
  return m_redo.back();
}

std::optional<JournalStep> Journal::takeRedo()
{
  if(m_redo.empty())
    return std::nullopt;
  m_undo.push_back(std::move(m_redo.back()));
  m_redo.pop_back();
  m_sealed = true;
  return m_undo.back();
}

void Journal::clear()
{
  m_undo.clear();
  m_redo.clear();
  m_sealed = false;
}

nlohmann::json Journal::topic() const
{
  nlohmann::json undo = nlohmann::json::array(), redo = nlohmann::json::array();
  for(auto it = m_undo.rbegin(); it != m_undo.rend(); ++it)
    undo.push_back(it->label);
  for(auto it = m_redo.rbegin(); it != m_redo.rend(); ++it)
    redo.push_back(it->label);
  return {{"undo", undo}, {"redo", redo}};
}

}  // namespace fv
