// A backdrop kit's list of plants (fv_host_backdrop.cpp): kit.json's, and kit.local.json's beside it,
// read the same way, each file named in what is wrong with it.

#include <gtest/gtest.h>

#include "fv_host_impl.hpp"

namespace {

std::string whyNot(const Json& kit, const std::string& name)
{
  try
  {
    fvhost::kitPlants(kit, "kit", name);
  }
  catch(const std::runtime_error& e)
  {
    return e.what();
  }
  return {};
}

}  // namespace

TEST(FvHostKit, APlantIsARoleAndAFileUnderTheKit)
{
  const auto plants = fvhost::kitPlants(Json::parse(R"({"version": 1, "plants": [
      {"role": "tree", "file": "fir/fir.gltf", "heaviest": 1600000},
      {"role": "broadleaf", "file": "blenderkit/alder/alder.gltf"}]})"),
                                        "kit", "kit.local.json");
  ASSERT_EQ(plants.size(), 2U);
  EXPECT_EQ(plants[0].role, "tree");
  EXPECT_EQ(plants[0].file, std::filesystem::path("kit") / "fir" / "fir.gltf");
  EXPECT_EQ(plants[0].heaviest, 1600000U);
  EXPECT_EQ(plants[1].role, "broadleaf");
  EXPECT_EQ(plants[1].heaviest, 0U);
  // a list with none is a kit with none of them, not a wrong one
  EXPECT_TRUE(fvhost::kitPlants(Json::parse(R"({"plants": []})"), "kit", "kit.local.json").empty());
}

TEST(FvHostKit, AWrongListSaysWhichFileItIs)
{
  EXPECT_NE(whyNot(Json::parse(R"({"version": 1})"), "kit.local.json").find("kit.local.json lists no plants"), std::string::npos);
  EXPECT_NE(whyNot(Json(), "kit.json").find("kit.json"), std::string::npos);  // (what a file that is not JSON parses to)
  EXPECT_NE(whyNot(Json::parse(R"({"plants": [{"role": "tree"}]})"), "kit.local.json").find("kit.local.json: a plant needs a role and a file"),
            std::string::npos);
  EXPECT_NE(whyNot(Json::parse(R"({"plants": [{"file": "a.gltf"}]})"), "kit.json").find("kit.json:"), std::string::npos);
}
