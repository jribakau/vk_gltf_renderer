#pragma once

// A DAZ animation made for one Genesis played on another: Genesis 8's walk on a Genesis 9. Their bones
// have other names (lShldrBend, l_upperarm), other axes, and limbs split otherwise (Genesis 8 bends
// and twists an arm on two bones, Genesis 9 on one), and they're built otherwise: a spine of other
// lengths, a jaw's or an eyelid's end placed elsewhere.
//
// So each key is rebuilt in the body's own space on the skeleton the file was made for (its base
// figure's, from the library): every bone's turn as its channels give it, through its parents. Each
// of the figure's limbs then turns from its own rest as its counterpart turns from the source's (a
// limb is a bone with its twist bones, matched by anatomy, wardrobe::parseBone), and its own channels
// are what turn it so (PoseRig::rotationTotals); a limb split in two takes the twist about its axis on
// its twist bone. How a bone points at rest is the figure's build, which it keeps: a face the file
// holds at rest stays the figure's own. Only a default pose apart re-aims a limb: Genesis 2 stands in
// a T, later ones in an A, so its arms are turned onto the source's, and what hangs from them with
// them. A bone between two matched limbs that the figure hasn't (Genesis 8's metatarsals) passes its
// turn on to the next. The hip's move is scaled by the two hips' heights.

#include <string>

#include <set>
#include <string>

#include <figure/dson/store.hpp>
#include <figure/pose/pose_clip.hpp>
#include <figure/pose/pose_rig.hpp>

namespace fv::pose {

// The base figure file (a library path) whose skeleton a pose file's bones are named for, by their
// names and then the file's own path (Genesis 8 Male's, Genesis 3's); empty when its bones are none
// of Genesis 2, 3, 8 or 9's.
std::string sourceFigureFile(const dson::Json& doc, const std::string& rel);
// The same from the bones a file names: their family by the skeleton spec (rig::identify), Genesis 2's,
// 3's and 8's sex and version by the path.
std::string sourceFigureFor(const std::set<std::string>& bones, const std::string& rel);
// That file's figure as people call it: "Genesis 8 Female".
std::string sourceFigureName(const std::string& file);

// Whether most of the bones the file names are this rig's: it plays as it is.
bool namesRigBones(const dson::Json& doc, const PoseRig& rig);

// The file's animation, made for `from`, as channels of `to`'s bones. Only the limbs the file moves.
PoseClip retargetClip(const dson::Json& doc, const PoseRig& from, const PoseRig& to);

}  // namespace fv::pose
