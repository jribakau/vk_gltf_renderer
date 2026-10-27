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

// The `tlasPolicy` setting: its names, and when a TLAS sync builds rather than updates.

#include <gtest/gtest.h>

#include "tlas_policy.hpp"

TEST(TlasPolicy, NamesRoundTrip)
{
  for(TlasPolicy policy : {TlasPolicy::eBuild, TlasPolicy::eUpdate})
    EXPECT_EQ(parseTlasPolicy(tlasPolicyName(policy)), policy);
  EXPECT_STREQ(tlasPolicyName(TlasPolicy::eBuild), "build");
  EXPECT_STREQ(tlasPolicyName(TlasPolicy::eUpdate), "update");
}

TEST(TlasPolicy, AnythingElseIsRejected)
{
  EXPECT_FALSE(parseTlasPolicy("").has_value());
  EXPECT_FALSE(parseTlasPolicy("Build").has_value());
  EXPECT_FALSE(parseTlasPolicy("rebuild").has_value());
  EXPECT_FALSE(parseTlasPolicy("update ").has_value());
}

TEST(TlasPolicy, BuildAlwaysBuilds)
{
  EXPECT_TRUE(tlasSyncBuilds(TlasPolicy::eBuild, false));
  EXPECT_TRUE(tlasSyncBuilds(TlasPolicy::eBuild, true));
}

// `update` refits, except when an instance became active or inactive: an UPDATE cannot bring an
// instance into the structure that was not in it
TEST(TlasPolicy, UpdateBuildsOnlyWhenAnInstanceChangedActiveState)
{
  EXPECT_FALSE(tlasSyncBuilds(TlasPolicy::eUpdate, false));
  EXPECT_TRUE(tlasSyncBuilds(TlasPolicy::eUpdate, true));
}
