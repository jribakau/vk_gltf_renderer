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

// The morph topic's list (publishMorphs): the shown dials as the panel is told of them, sent whole once
// and then as what changed in them, nothing for a change to hidden channels alone.

#include <gtest/gtest.h>

#include "fv_host_impl.hpp"

namespace {

fv::morph::Loader nothing()
{
  return []() -> fv::morph::Finish { return [] { return fv::morph::LazyResult{}; }; };
}

// dials with nothing in their files, in this order
struct Dials
{
  fv::morph::MorphSet set{[](int) -> fv::mesh::BuiltMesh* { return nullptr; }};
  Dials(std::initializer_list<const char*> names)
  {
    for(const char* name : names)
      set.addLazy(name, nothing());
  }
};

std::vector<ListedDial> listOf(std::initializer_list<const char*> names)
{
  return listedDials(Dials(names).set);
}

}  // namespace

// A dial's first drag brings its correctives in hidden: the list's version moves, the list doesn't.
TEST(MorphList, HiddenChannelsJoiningSendNothing)
{
  Dials                         dials{"Smile", "Waist"};
  fv::morph::MorphSet&          set  = dials.set;
  const std::vector<ListedDial> sent = listedDials(set);
  const int                     was  = set.listVersion();
  fv::morph::ChannelOptions     hidden;
  hidden.hidden = true;
  set.addLazy("pJCMShldrUp_40_L", nothing(), hidden);
  EXPECT_NE(set.listVersion(), was);

  const std::optional<Json> patch = listedPatch(sent, listedDials(set));
  ASSERT_TRUE(patch.has_value());
  EXPECT_TRUE(patch->empty()) << patch->dump();
}

// What changed in a dial goes as a merge patch of it alone: a field it gained, a field it lost as null.
TEST(MorphList, AChangedDialGoesAsWhatChangedInIt)
{
  std::vector<ListedDial> sent = listOf({"Smile", "Waist"});
  sent[1].path                 = "/Old";
  std::vector<ListedDial>       now  = sent;
  now[0].items                       = {"Shirt"};
  now[1].path.clear();
  now[1].reach = ListedDial::Reach{1.5, 0, 2, 0, 0};

  const std::optional<Json> patch = listedPatch(sent, now);
  ASSERT_TRUE(patch.has_value());
  const Json want = {{"Smile", {{"items", {"Shirt"}}}},
                     {"Waist", {{"path", nullptr}, {"reach", {{"cm", 1.5}, {"deg", 0.0}, {"morphs", 2}, {"bones", 0}, {"scales", 0}}}}}};
  EXPECT_EQ(*patch, want) << patch->dump();
}

// Dials that came, went or changed places: the whole list goes again, as each one's order changed.
TEST(MorphList, DialsComingGoingOrMovingSendTheWholeList)
{
  const std::vector<ListedDial> two = listOf({"Smile", "Waist"});
  EXPECT_FALSE(listedPatch(two, listOf({"Smile", "Waist", "Chin"})).has_value());
  EXPECT_FALSE(listedPatch(two, listOf({"Smile"})).has_value());
  EXPECT_FALSE(listedPatch(two, listOf({"Waist", "Smile"})).has_value());
}

// The whole list as the panel reads it: each dial in its order, a field only where it says something.
TEST(MorphList, TheWholeListSaysEachDialsPlaceAndWhatItHas)
{
  std::vector<ListedDial> dials = listOf({"Smile", "Waist"});
  dials[1].helper               = true;
  dials[1].reach                = ListedDial::Reach{0.123, 4.5, 1, 2, 3};
  const Json channels           = listedChannels(dials);
  ASSERT_EQ(channels.size(), 2u);
  EXPECT_EQ(channels.at("Smile").value("order", -1), 0);
  EXPECT_EQ(channels.at("Waist").value("order", -1), 1);
  EXPECT_EQ(channels.at("Smile").value("label", ""), dials[0].label);
  EXPECT_FALSE(channels.at("Smile").contains("helper"));
  EXPECT_FALSE(channels.at("Smile").contains("items"));
  EXPECT_FALSE(channels.at("Smile").contains("reach"));
  EXPECT_TRUE(channels.at("Waist").value("helper", false));
  EXPECT_EQ(channels.at("Waist").at("reach").value("cm", 0.0), 0.123);
  for(const char* key : {"label", "group", "value", "min", "max", "weight", "order"})
    EXPECT_TRUE(channels.at("Smile").contains(key)) << key;
}
