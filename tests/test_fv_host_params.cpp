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

// A batch of renderer parameters as the host sets it (`params.set`, setExposedParameters): a value the
// batch names holds against another parameter's callback.

#include <gtest/gtest.h>

#include "fv_host_impl.hpp"

namespace {

// The two parameters of a still's samples per frame, as PathTracer::registerParameters declares them:
// setting ptSamples by hand turns Auto SPP off, and ptAdaptiveSampling is a plain value.
struct SamplesParameters
{
  SamplesParameters()
  {
    registry.add({.name = "ptSamples", .callbackSuccess = [this](const nvutils::ParameterBase* const) { autoSpp = false; }},
                 &samples, 1, 100);
    registry.add({.name = "ptAdaptiveSampling"}, &autoSpp);
  }
  Json set(const Json& values) { return fvhost::setExposedParameters(&registry, Json{{"values", values}}); }

  nvutils::ParameterRegistry registry;
  int                        samples = 1;
  bool                       autoSpp = true;
};

// What the answer says `name` is now
Json updatedValue(const Json& answer, const std::string& name)
{
  for(const Json& p : answer.value("updated", Json::array()))
    if(p.value("name", "") == name)
      return p.value("value", Json());
  return Json();
}

}  // namespace

// The app's RTX batch (rtx.ts rendering()) names both. Every value is written before any callback
// runs, so ptSamples' callback turned off the Auto SPP the same batch asked for: a still in the app
// traced one sample a frame, whatever the batch said.
TEST(FvHostParams, AutoSppNamedBesideSamplesStaysOn)
{
  SamplesParameters p;
  const Json answer = p.set({{"ptSamples", 1}, {"ptAdaptiveSampling", true}});
  EXPECT_TRUE(p.autoSpp);
  EXPECT_EQ(p.samples, 1);
  // and the answer says what it is, not what the callback left
  EXPECT_EQ(updatedValue(answer, "ptAdaptiveSampling"), Json(true));

  // sent again, as the app does on every Render tab change
  p.set({{"ptSamples", 1}, {"ptAdaptiveSampling", true}});
  EXPECT_TRUE(p.autoSpp);
}

// A script that fixes the count names ptSamples alone, or Auto SPP off beside it: both as before
TEST(FvHostParams, SamplesOnTheirOwnStillFixTheCount)
{
  SamplesParameters p;
  p.set({{"ptSamples", 4}});
  EXPECT_FALSE(p.autoSpp);
  EXPECT_EQ(p.samples, 4);

  p.autoSpp = true;
  const Json answer = p.set({{"ptSamples", 8}, {"ptAdaptiveSampling", false}});
  EXPECT_FALSE(p.autoSpp);
  EXPECT_EQ(p.samples, 8);
  EXPECT_EQ(updatedValue(answer, "ptAdaptiveSampling"), Json(false));
}

// A parameter with a callback of its own is left as that callback left it: it may clamp its value
TEST(FvHostParams, AParameterWithACallbackKeepsWhatItsCallbackLeft)
{
  nvutils::ParameterRegistry registry;
  int                        budget = 8;
  registry.add({.name = "budget", .callbackSuccess = [&](const nvutils::ParameterBase* const) { budget = std::max(budget, 1); }},
               &budget);
  const Json answer = fvhost::setExposedParameters(&registry, Json{{"values", {{"budget", 0}}}});
  EXPECT_EQ(budget, 1);
  EXPECT_EQ(updatedValue(answer, "budget"), Json(1));
}

// A batch is still refused whole when it names a parameter this build lacks, with the message the app
// reads the name from (rtx.ts RtxQuality.set)
TEST(FvHostParams, AnUnknownNameRefusesTheBatchWhole)
{
  SamplesParameters p;
  try
  {
    p.set({{"ptSamples", 4}, {"ptAdaptiveSampling", false}, {"noSuchParameter", 1}});
    FAIL() << "the batch was taken";
  }
  catch(const std::exception& e)
  {
    EXPECT_NE(std::string(e.what()).find("noSuchParameter"), std::string::npos) << e.what();
  }
  EXPECT_EQ(p.samples, 1);
  EXPECT_TRUE(p.autoSpp);
}
