# Figure Viewer scene logic: DAZ figures, morphs, posing, wardrobe. No Vulkan, so it builds into
# the renderer and into the unit tests alike. Headers are included as <figure/...>.
file(GLOB_RECURSE FV_FIGURE_SOURCES CONFIGURE_DEPENDS "*.cpp" "*.hpp")
list(FILTER FV_FIGURE_SOURCES EXCLUDE REGEX "/tools/")

add_library(fv_figure STATIC ${FV_FIGURE_SOURCES})
target_compile_features(fv_figure PUBLIC cxx_std_20)
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH FV_FIGURE_ROOT)
target_include_directories(fv_figure PUBLIC ${FV_FIGURE_ROOT})
# tinygltf for the glTF model it builds and for nlohmann::json (<tinygltf/json.hpp>); meshoptimizer for
# the coarse copies of colliders the cloth meets (physics/cloth/coarse.cpp)
target_link_libraries(fv_figure PUBLIC glm tinygltf PRIVATE zlib meshoptimizer)
set_property(TARGET fv_figure PROPERTY FOLDER "figure")
# The bind cache's key (wardrobe/bind_cache.cpp): xxHash as zstd ships it, and the hash of the binding
# code (bind.cpp and the closest-point search it runs on), so a change to it is another key. Those
# files are configure dependencies for that.
target_include_directories(fv_figure PRIVATE ${nvpro_core2_SOURCE_DIR}/third_party/zstd/lib/common)
set(FV_BIND_SOURCES wardrobe/bind.cpp physics/collide/surface_index.hpp physics/collide/surface_index.cpp)
set(FV_BIND_SOURCE_TEXT "")
foreach(source ${FV_BIND_SOURCES})
  file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/${source} sourceSha)
  string(APPEND FV_BIND_SOURCE_TEXT "${sourceSha}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source})
endforeach()
string(SHA256 FV_BIND_SOURCE_SHA "${FV_BIND_SOURCE_TEXT}")
set_source_files_properties(wardrobe/bind_cache.cpp PROPERTIES COMPILE_DEFINITIONS "FV_BIND_SOURCE_SHA=\"${FV_BIND_SOURCE_SHA}\"")
# The body map cache's key (bodymap/cache.cpp) likewise: the map code, the skeleton spec its regions
# come from, and the closest-point search.
file(GLOB FV_BODYMAP_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS bodymap/*.cpp bodymap/*.hpp rig/*.cpp rig/*.hpp rig/aliases/*.cpp)
list(SORT FV_BODYMAP_SOURCES)
list(APPEND FV_BODYMAP_SOURCES physics/collide/surface_index.hpp physics/collide/surface_index.cpp)
set(FV_BODYMAP_SOURCE_TEXT "")
foreach(source ${FV_BODYMAP_SOURCES})
  file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/${source} sourceSha)
  string(APPEND FV_BODYMAP_SOURCE_TEXT "${sourceSha}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source})
endforeach()
string(SHA256 FV_BODYMAP_SOURCE_SHA "${FV_BODYMAP_SOURCE_TEXT}")
set_source_files_properties(bodymap/cache.cpp PROPERTIES COMPILE_DEFINITIONS "FV_BODYMAP_SOURCE_SHA=\"${FV_BODYMAP_SOURCE_SHA}\"")

# The parity harness's native side (see tools/fv_parity.cpp): no GPU. The library, and for its feed
# step the retained scene's Vulkan-free sources: the store, the figure feed, and what they call (the
# importer's tangent rule, the material feature detector, the tangent helpers, which use nvutils'
# logger and pool and nvshaders' shared functions), with tinygltf's out-of-line half.
add_executable(fv_parity tools/fv_parity.cpp tools/fv_parity_tinygltf.cpp tools/material_sweep.cpp tools/material_views.cpp tools/closure_probe.cpp
  ${FV_FIGURE_ROOT}/src/gltf_material_cache.cpp
  ${FV_FIGURE_ROOT}/src/scene_retained.cpp
  ${FV_FIGURE_ROOT}/src/fv_figure_feed.cpp
  ${FV_FIGURE_ROOT}/src/scene_import_gltf.cpp
  ${FV_FIGURE_ROOT}/src/ies_profile.cpp
  ${FV_FIGURE_ROOT}/src/scene_feature_detection.cpp
  ${FV_FIGURE_ROOT}/src/tinygltf_utils.cpp)
target_include_directories(fv_parity PRIVATE ${FV_FIGURE_ROOT}/src ${NVSHADERS_DIR})
target_link_libraries(fv_parity PRIVATE fv_figure nvpro2::nvutils)
set_target_properties(fv_parity PROPERTIES FOLDER "figure" RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/_bin/Release
                                           RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/_bin/Debug)
