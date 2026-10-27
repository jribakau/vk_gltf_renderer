# Figure Viewer scene logic: DAZ figures, morphs, posing, wardrobe. No Vulkan, so it builds into
# the renderer and into the unit tests alike. Headers are included as <figure/...>.
file(GLOB_RECURSE FV_FIGURE_SOURCES CONFIGURE_DEPENDS "*.cpp" "*.hpp")
list(FILTER FV_FIGURE_SOURCES EXCLUDE REGEX "/tools/")

add_library(fv_figure STATIC ${FV_FIGURE_SOURCES})
target_compile_features(fv_figure PUBLIC cxx_std_20)
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH FV_FIGURE_ROOT)
target_include_directories(fv_figure PUBLIC ${FV_FIGURE_ROOT})
# tinygltf for the glTF model it builds and for nlohmann::json (<tinygltf/json.hpp>)
target_link_libraries(fv_figure PUBLIC glm tinygltf PRIVATE zlib)
set_property(TARGET fv_figure PROPERTY FOLDER "figure")
# The bind cache's key (wardrobe/bind_cache.cpp): xxHash as zstd ships it, and bind.cpp's hash, so a
# change to the binding code is another key. bind.cpp is a configure dependency for that.
target_include_directories(fv_figure PRIVATE ${nvpro_core2_SOURCE_DIR}/third_party/zstd/lib/common)
file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/wardrobe/bind.cpp FV_BIND_SOURCE_SHA)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/wardrobe/bind.cpp)
set_source_files_properties(wardrobe/bind_cache.cpp PROPERTIES COMPILE_DEFINITIONS "FV_BIND_SOURCE_SHA=\"${FV_BIND_SOURCE_SHA}\"")

# The parity harness's native side (see tools/fv_parity.cpp): no GPU. The library, and for its feed
# step the retained scene's Vulkan-free sources: the store, the figure feed, and what they call (the
# importer's tangent rule, the material feature detector, the tangent helpers, which use nvutils'
# logger and pool and nvshaders' shared functions), with tinygltf's out-of-line half.
add_executable(fv_parity tools/fv_parity.cpp tools/fv_parity_tinygltf.cpp
  ${FV_FIGURE_ROOT}/src/scene_retained.cpp
  ${FV_FIGURE_ROOT}/src/fv_figure_feed.cpp
  ${FV_FIGURE_ROOT}/src/scene_import_gltf.cpp
  ${FV_FIGURE_ROOT}/src/scene_feature_detection.cpp
  ${FV_FIGURE_ROOT}/src/tinygltf_utils.cpp)
target_include_directories(fv_parity PRIVATE ${FV_FIGURE_ROOT}/src ${NVSHADERS_DIR})
target_link_libraries(fv_parity PRIVATE fv_figure nvpro2::nvutils)
set_target_properties(fv_parity PROPERTIES FOLDER "figure" RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/_bin/Release
                                           RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/_bin/Debug)
