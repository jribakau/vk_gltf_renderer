// See material_views.hpp.

#include "material_views.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// stb's writer kept to this file, as export_glb.cpp keeps its own; tinygltf as fv_parity_tinygltf.cpp
// builds it, without images
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include <tinygltf/tiny_gltf.h>

#include <figure/figure.hpp>

#include "closure_row.hpp"

namespace fv::probe {

using dson::Json;

namespace {

// A surface's name as a folder's: letters, digits, '-' and '_' kept, the rest '_'
std::string folderName(const std::string& name)
{
  std::string out;
  for(const char c : name)
    out += (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') ? c : '_';
  return out.empty() ? std::string("surface") : out;
}

void writePng(const std::filesystem::path& file, int w, int h, const std::vector<unsigned char>& rgb)
{
  std::ofstream out(file, std::ios::binary);
  stbi_write_png_to_func([](void* ctx, void* data, int size) { static_cast<std::ofstream*>(ctx)->write(static_cast<const char*>(data), size); },
                         &out, w, h, 3, rgb.data(), w * 3);
}

// Portable float map: linear RGB, rows from the bottom, little-endian
void writePfm(const std::filesystem::path& file, const ViewImage& image)
{
  std::ofstream out(file, std::ios::binary);
  out << "PF\n" << image.width << " " << image.height << "\n-1.0\n";
  for(int y = image.height - 1; y >= 0; y--)
    out.write(reinterpret_cast<const char*>(image.at(0, y)), std::streamsize(sizeof(float)) * 3 * image.width);
}

// The pictures side by side in rows, on dark grey
std::vector<unsigned char> sheet(const std::vector<std::vector<const std::vector<unsigned char>*>>& rows, int size, int& w, int& h)
{
  size_t columns = 1;
  for(const auto& row : rows)
    columns = std::max(columns, row.size());
  w = int(columns) * size;
  h = int(rows.size()) * size;
  std::vector<unsigned char> px(size_t(w) * h * 3, 40);
  for(size_t r = 0; r < rows.size(); r++)
    for(size_t c = 0; c < rows[r].size(); c++)
      for(int y = 0; y < size; y++)
        std::copy_n(rows[r][c]->data() + size_t(y) * size * 3, size_t(size) * 3,
                    px.data() + ((r * size + y) * size_t(w) + c * size) * 3);
  return px;
}

tinygltf::Material gltfMaterial(const Json& material)
{
  const Json         doc = {{"asset", {{"version", "2.0"}}}, {"materials", Json::array({material})}};
  const std::string  text = doc.dump();
  tinygltf::TinyGLTF loader;
  tinygltf::Model    model;
  std::string        err, warn;
  if(!loader.LoadASCIIFromString(&model, &err, &warn, text.c_str(), unsigned(text.size()), "") || model.materials.empty())
    throw std::runtime_error("slice: the glTF material doesn't parse: " + err);
  return model.materials[0];
}

}  // namespace

Json materialViews(const std::filesystem::path& libraryDir, const std::filesystem::path& root, const Json& step, const std::vector<Material>* loaded)
{
  if(!step.contains("out"))
    throw std::runtime_error("slice: no \"out\" folder");
  const std::filesystem::path out  = root / step["out"].get<std::string>();
  const mat::SkinMode         skin = skinNamed(step.value("skin", std::string("measured")));
  ViewOptions                 views;
  views.sliceSize = views.ballSize = step.value("size", 256);

  // What to draw: a glTF material, a file's surfaces, or the loaded figure's
  std::vector<std::pair<std::string, tinygltf::Material>> materials;
  std::unique_ptr<Figure>                                  figure;
  const std::vector<Material>*                             surfaces = loaded;
  if(step.contains("gltf"))
    materials.emplace_back(step["gltf"].value("name", std::string("material")), gltfMaterial(step["gltf"]));
  else
  {
    if(step.contains("file"))
    {
      auto store = std::make_shared<dson::Store>(std::make_shared<dson::Library>(std::vector<std::filesystem::path>{libraryDir}));
      figure     = Figure::load(store, step["file"].get<std::string>(), {.subdivision = 0});
      surfaces   = &figure->loaded.scene.materials;
    }
    if(!surfaces)
      throw std::runtime_error("slice: no \"file\" or \"gltf\", and no figure loaded");
    std::set<std::string> wanted;
    for(const Json& name : step.value("surfaces", Json::array()))
      wanted.insert(name.get<std::string>());
    std::set<std::string> names;
    for(const Material& m : *surfaces)
      if((wanted.empty() || wanted.count(m.name)) && names.insert(m.name).second)
        materials.emplace_back(m.name, loweredGltf(m, skin));
    if(materials.empty())
      throw std::runtime_error("slice: none of the surfaces asked for");
  }

  Json report = Json::object();
  for(const auto& [name, material] : materials)
  {
    const std::filesystem::path dir = out / folderName(name);
    std::filesystem::create_directories(dir);
    const MaterialViews v    = viewsOfGltf(material, {}, views);
    const int           size = views.sliceSize;

    const auto litPx     = ballPixels(v.lit);
    const auto furnacePx = ballPixels(v.furnace);
    const auto totalPx   = slicePixels(v.sliceTotal);
    writePng(dir / "lit.png", size, size, litPx);
    writePng(dir / "furnace.png", size, size, furnacePx);
    writePng(dir / "slice.png", size, size, totalPx);
    writePfm(dir / "lit.pfm", v.lit);
    writePfm(dir / "furnace.pfm", v.furnace);
    writePfm(dir / "slice.pfm", v.sliceTotal);

    std::vector<std::vector<unsigned char>> lobePx;
    Json                                    lobes = Json::array();
    for(int lobe = 0; lobe < kLobeCount; lobe++)
    {
      if(!v.lobeUsed[lobe])
        continue;
      lobes.push_back(lobeName(lobe));
      lobePx.push_back(slicePixels(v.slice[lobe]));
      const std::string stem = std::string("slice_") + lobeName(lobe);
      writePng(dir / (stem + ".png"), size, size, lobePx.back());
      writePfm(dir / (stem + ".pfm"), v.slice[lobe]);
    }
    std::vector<const std::vector<unsigned char>*> lobeRow;
    for(const auto& px : lobePx)
      lobeRow.push_back(&px);
    std::vector<std::vector<const std::vector<unsigned char>*>> rows{{&litPx, &furnacePx, &totalPx}};
    if(!lobeRow.empty())
      rows.push_back(lobeRow);
    int        w = 0, h = 0;
    const auto px = sheet(rows, size, w, h);
    writePng(dir / "sheet.png", w, h, px);

    // the albedo head-on and near grazing, as the furnace ball shows it at its centre and by its rim
    const float* faceOn  = v.furnace.at(size / 2, size / 2);
    const float* grazing = v.furnace.at(size / 2, std::max(1, size / 64));
    const std::u8string folder = dir.generic_u8string();
    report[name] = {{"folder", std::string(folder.begin(), folder.end())},
                    {"lobes", lobes},
                    {"albedoFaceOn", {faceOn[0], faceOn[1], faceOn[2]}},
                    {"albedoGrazing", {grazing[0], grazing[1], grazing[2]}}};
    std::fprintf(stderr, "[slice] %s -> %s\n", name.c_str(), std::string(folder.begin(), folder.end()).c_str());
  }
  return report;
}

}  // namespace fv::probe
