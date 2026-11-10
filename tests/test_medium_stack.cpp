// The media a path is inside (shaders/medium_stack.h.slang) compiled as C++: a straight line through
// solids that contain, overlap or only stand near one another crosses each boundary once, with the
// right medium on either side, and ends outside.

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../shaders/medium_stack.h.slang"

namespace {

constexpr int kGlass = 3, kDrink = 5, kIce = 7, kSkin = 9, kSpoon = 11;

struct Face
{
  int  material;
  bool front;
  bool bounds = true;
};

// The boundaries a ray crosses going straight through `faces`, as "a|b" (-1: air), and "." for a face
// passed as no boundary; `state` is left as the ray ends
std::vector<std::string> crossings(const std::vector<Face>& faces, MediumState& state, bool walk = false)
{
  std::vector<std::string> out;
  for(const Face& f : faces)
  {
    const MediumBoundary b = mediumBoundary(state, f.material, f.bounds, f.front, walk);
    if(b.passed)
    {
      out.push_back(".");
      state = b.here;
      continue;
    }
    const int own = f.bounds ? f.material : -2;  // -2: a surface with no medium of its own
    out.push_back(b.ownSide ? std::to_string(own) + "|" + std::to_string(b.against) : std::to_string(b.against) + "|" + std::to_string(own));
    state = b.beyond;
    walk  = false;  // a walk is the first state's only
  }
  return out;
}

using Crossings = std::vector<std::string>;

}  // namespace

TEST(MediumStack, ASolidIsEnteredAndLeft)
{
  MediumState s = mediumOutside();
  EXPECT_EQ(crossings({{kGlass, true}, {kGlass, false}}, s), (Crossings{"-1|3", "3|-1"}));
  EXPECT_FALSE(s.inside);
}

// Oil in a bottle that is one solid of glass: the oil's faces are met inside the glass
TEST(MediumStack, ALiquidInsideASolidMeetsItNotAir)
{
  MediumState s = mediumOutside();
  EXPECT_EQ(crossings({{kGlass, true}, {kDrink, true}, {kDrink, false}, {kGlass, false}}, s), (Crossings{"-1|3", "3|5", "5|3", "3|-1"}));
  EXPECT_FALSE(s.inside);
  EXPECT_EQ(s.current, -1);
}

// A drink modelled a little into its glass's wall: the wall's inner face lies inside the drink, and
// is no boundary. Each side of the glass is crossed once into the drink and once out of it.
TEST(MediumStack, WhereTwoSolidsOverlapTheFaceInsideTheOtherIsNoBoundary)
{
  MediumState s = mediumOutside();
  // through the side: outer wall, the drink's face (in the wall), the wall's inner face (in the drink),
  // then the far side: the inner face, the drink's face, the outer wall
  EXPECT_EQ(crossings({{kGlass, true}, {kDrink, true}, {kGlass, false}, {kGlass, true}, {kDrink, false}, {kGlass, false}}, s),
            (Crossings{"-1|3", "3|5", ".", "5|3", ".", "3|-1"}));
  EXPECT_FALSE(s.inside);

  // from above: the drink's own surface first, then down through the bottom
  s = mediumOutside();
  EXPECT_EQ(crossings({{kDrink, true}, {kGlass, true}, {kDrink, false}, {kGlass, false}}, s), (Crossings{"-1|5", "5|3", ".", "3|-1"}));
  EXPECT_FALSE(s.inside);
}

// A drink that stops short of its glass: air between them, and each face against air
TEST(MediumStack, AGapBetweenTwoSolidsIsAir)
{
  MediumState s = mediumOutside();
  EXPECT_EQ(crossings({{kGlass, true}, {kGlass, false}, {kDrink, true}, {kDrink, false}, {kGlass, true}, {kGlass, false}}, s),
            (Crossings{"-1|3", "3|-1", "-1|5", "5|-1", "-1|3", "3|-1"}));
  EXPECT_FALSE(s.inside);
}

// The audit's re-entry: the back of a solid met from outside (the path started in it, or its mesh is
// open) is that solid's way out, and the path is outside after it, not inside
TEST(MediumStack, ABackFaceMetOutsideLeavesTheSolidItBelongsTo)
{
  MediumState          s = mediumOutside();
  const MediumBoundary b = mediumBoundary(s, kGlass, true, false, false);
  EXPECT_TRUE(b.ownSide);
  EXPECT_EQ(b.against, -1);
  EXPECT_FALSE(b.beyond.inside);
  EXPECT_FALSE(b.passed);
}

// An unknown solid's back met inside a known one (two materials on one bottle): the path leaves what
// it was in, as it always did
TEST(MediumStack, AnotherMaterialsBackFaceLeavesTheCurrentMedium)
{
  MediumState s = mediumEntered(mediumOutside(), kGlass);
  EXPECT_EQ(crossings({{kDrink, false}}, s), (Crossings{"5|-1"}));
  EXPECT_FALSE(s.inside);
}

// Ice in a drink in a solid glass: three deep, one more than is kept. The innermost boundaries are
// right; the glass is forgotten where the ice is met, so the drink is left into air, and the glass's
// back still takes the path out and no further.
TEST(MediumStack, AThirdMediumDropsTheOutermost)
{
  MediumState s = mediumOutside();
  EXPECT_EQ(crossings({{kGlass, true}, {kDrink, true}, {kIce, true}, {kIce, false}, {kDrink, false}, {kGlass, false}}, s),
            (Crossings{"-1|3", "3|5", "5|7", "7|5", "5|-1", "3|-1"}));
  EXPECT_FALSE(s.inside);
}

// The medium around is set where the inner surface is met, so a path that bounces off the ice has
// forgotten the glass as one that went through it has; a path that bounces off the drink's own face,
// inside the glass, is still in the glass alone
TEST(MediumStack, TheMediumAroundIsSetWhereTheInnerSurfaceIsMet)
{
  // in the glass, at the drink's face: here the glass is what is around the drink, and nothing is around the glass
  MediumState          glass = mediumEntered(mediumOutside(), kGlass);
  const MediumBoundary face  = mediumBoundary(glass, kDrink, true, true, false);
  EXPECT_EQ(face.here.current, kGlass);
  EXPECT_EQ(face.here.outer, kGlass);
  EXPECT_FALSE(mediumLeft(face.here).inside) << "bounced back: leaving the glass is leaving every medium";
  EXPECT_EQ(face.beyond.current, kDrink);
  EXPECT_EQ(face.beyond.outer, kGlass);
  EXPECT_EQ(mediumLeft(face.beyond).current, kGlass);

  // in the drink (in the glass), bounced off the ice: the glass is forgotten
  const MediumBoundary ice = mediumBoundary(face.beyond, kIce, true, true, false);
  MediumState          s   = ice.here;
  EXPECT_EQ(crossings({{kDrink, false}, {kGlass, false}}, s), (Crossings{"5|-1", "3|-1"}));
  EXPECT_FALSE(s.inside);
}

// A spoon in the drink bounds no medium: it is met in the drink, and the path stays there
TEST(MediumStack, ASurfaceWithoutAMediumIsMetInTheOneAroundIt)
{
  MediumState s = mediumEntered(mediumOutside(), kDrink);
  for(bool front : {true, false})
  {
    const MediumBoundary b = mediumBoundary(s, kSpoon, false, front, false);
    EXPECT_EQ(b.against, kDrink);
    EXPECT_FALSE(b.ownSide);
    EXPECT_FALSE(b.passed);
    EXPECT_EQ(b.beyond.current, kDrink);
    EXPECT_TRUE(b.beyond.inside);
    EXPECT_EQ(b.here.outer, s.outer) << "nothing of its own to be around";
  }
}

// A walk through skin ends at any front face: an opening it escaped through, or what sits in one.
// In a tub, it is back in the water it entered the skin from.
TEST(MediumStack, AWalkEndsAtAFrontFace)
{
  MediumState          s = mediumEntered(mediumOutside(), kSkin);
  const MediumBoundary b = mediumBoundary(s, kSpoon, false, true, /*walk=*/true);
  EXPECT_FALSE(b.here.inside);
  EXPECT_EQ(b.against, -1);

  // the skin entered from the water: through its face as the water met it
  s                      = mediumBoundary(mediumEntered(mediumOutside(), kDrink), kSkin, true, true, false).beyond;
  EXPECT_EQ(s.current, kSkin);
  EXPECT_EQ(s.outer, kDrink);
  const MediumBoundary w = mediumBoundary(s, kSpoon, false, true, /*walk=*/true);
  EXPECT_TRUE(w.here.inside);
  EXPECT_EQ(w.here.current, kDrink);
  EXPECT_EQ(w.against, kDrink);

  // its own surface from inside is its way out, into the water
  const MediumBoundary out = mediumBoundary(s, kSkin, true, false, /*walk=*/true);
  EXPECT_TRUE(out.ownSide);
  EXPECT_EQ(out.against, kDrink);
  EXPECT_EQ(out.beyond.current, kDrink);
}

// A pane that is one sheet of faces with Thin Walled off never lets the path out. The next sheet of
// the same glass shows the path was not in it: it is entered from air, not from glass.
TEST(MediumStack, TheFrontOfTheMediumThePathThinksItIsInSaysItIsNot)
{
  MediumState s = mediumOutside();
  EXPECT_EQ(crossings({{kGlass, true}, {kGlass, true}, {kGlass, true}}, s), (Crossings{"-1|3", "-1|3", "-1|3"}));
  EXPECT_EQ(s.current, kGlass);
  EXPECT_EQ(s.outer, -1);
}

// The camera's medium has no material: a solid entered in it is left into no medium, as before
TEST(MediumStack, TheCamerasMediumHasNoName)
{
  MediumState s;
  s.inside  = true;
  s.current = -1;
  s.outer   = -1;
  EXPECT_EQ(crossings({{kGlass, true}, {kGlass, false}}, s), (Crossings{"-1|3", "3|-1"}));
  EXPECT_FALSE(s.inside);
}
