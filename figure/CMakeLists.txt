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

# The parity harness's native side (see tools/fv_parity.cpp): no GPU, the library alone.
add_executable(fv_parity tools/fv_parity.cpp)
target_link_libraries(fv_parity PRIVATE fv_figure)
set_target_properties(fv_parity PROPERTIES FOLDER "figure" RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/_bin/Release
                                           RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/_bin/Debug)
