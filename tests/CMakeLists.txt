# Test executable
set(TEST_NAME ${PROJECT_NAME}_tests)

# Common test utilities
set(TEST_COMMON_SOURCES
    common/test_utils.hpp
    common/test_utils.cpp
    common/fake_gpu_backend.hpp
    common/fake_gpu_backend.cpp
    common/residency_oracle.hpp
    common/residency_oracle.cpp
    test_main.cpp
)

# Test sources (will grow in each phase)
set(TEST_SOURCES
    ${TEST_COMMON_SOURCES}
    test_basic.cpp
    # Phase 2A: Validation and round-trip tests
    test_roundtrip.cpp
    test_features_preserved.cpp
    # Phase 2B: Model-primary architecture tests
    test_model_primary.cpp
    # Phase 2C: Basic editing operations tests
    test_basic_editing.cpp
    # Phase 2D.1: Basic index remapping tests
    test_index_remapping_basic.cpp
    # Phase 2D.2: Advanced index remapping tests
    test_index_remapping_advanced.cpp
    # Dirty flags and RenderNodes/RenderPrimitives
    test_dirty_flags_and_render_nodes.cpp
    # Node hierarchy operations (add, duplicate, reparent, delete)
    test_node_hierarchy_operations.cpp
    # Children order preservation test
    test_children_order_preservation.cpp
    # Animation and validation (empty sampler, compact, validator)
    test_animation_and_validation.cpp
    # Error paths and edge cases (bounds checks, merger failures, load failures)
    test_error_paths.cpp
    # MaterialCache conversion, incremental updates, topology change detection
    test_material_cache.cpp
    # Animation update, time advancement, wrapping, stability
    test_animation_update.cpp
    # Compute animation: deferred CPU allocation, CPU skinning, CPU morph targets
    test_compute_animation.cpp
    # Top-level extensionsUsed / extensionsRequired reconciliation on save
    test_extensions_metadata.cpp
    # KHR_materials_scatter host-side contract (defaults, clamping, legacy alias, round-trip)
    test_materials_scatter.cpp
    # KHR_materials_diffuse_transmission property names (spec key, legacy fallback, round-trip)
    test_materials_diffuse_transmission.cpp
    # Procedural primitives (plane/cube/sphere) + new/empty-scene workflow
    test_primitives.cpp
    # Tangents made side by side match the serial loop bit for bit
    test_tangents.cpp
    # Agentic workflow bridge contract
    test_agentic_bridge.cpp
    # Geometry compaction preserves EXT_mesh_opacity_micromap payload
    test_opacity_micromap_compaction.cpp
    # KHR_interactivity behavior-graph engine (parsing, flow scheduling, node evaluation)
    test_interactivity_engine.cpp
    # KHR_interactivity Phase C: pointer/get, pointer/set against a real Scene
    test_interactivity_pointer.cpp
    # KHR_interactivity Phase E: event/onSelect, event/onHoverIn/onHoverOut bubbling
    test_interactivity_hover_select.cpp
    # AnimationPointerSystem: partial texture-extension writes preserve sibling fields
    test_animation_pointer.cpp
    # glTF 2.1 conformance against the generated fixtures in gltf_2_1/
    test_gltf_2_1.cpp
    # glTF 2.1 uid: copies, merge collisions, validator
    test_uid.cpp
    # Image decoding (WebP path of the shared loader)
    test_image_loader.cpp
    # Phase 1 (env pipeline): CPU-only unit test for nvvk::buildEnvAliasmap (Vose)
    test_alias_table.cpp
    # --stream frame pacing: paced, except a still being refined
    test_frame_pacing.cpp
    # Binary payloads posted for live edits (/blob): taken once, oldest dropped past the cap
    test_blob_store.cpp
    # Figure Viewer scene logic (figure/)
    figure/test_gzip.cpp
    figure/test_dson_parse.cpp
    figure/test_jobs.cpp
    figure/test_catmark.cpp
    figure/test_formulas.cpp
    figure/test_erc.cpp
    figure/test_morph_set.cpp
    figure/test_figure_driver.cpp
    figure/test_eyes.cpp
    figure/test_pose_gizmo.cpp
    figure/test_presets.cpp
    figure/test_materials.cpp
    figure/test_wardrobe.cpp
    figure/test_bind_cache.cpp
    figure/test_hair_strands.cpp
    figure/test_collider.cpp
    figure/test_export.cpp
    figure/test_dforce.cpp
    figure/test_scene_load.cpp
    figure/test_skin.cpp
    # Figure Viewer host protocol: commands and the event stream, without a server
    test_fv_protocol.cpp
    ${CMAKE_SOURCE_DIR}/src/fv_events.cpp
    # Figure Viewer scene of props composed into the renderer's one glTF
    test_fv_compose.cpp
    ${CMAKE_SOURCE_DIR}/src/fv_compose.cpp
    # A model prop's material with its edits, for the scene and for an export
    test_fv_model_material.cpp
    ${CMAKE_SOURCE_DIR}/src/fv_model_material.cpp
    # A prop's nodes topic: taken-off items left out, a model's own names
    test_fv_view_describe.cpp
    ${CMAKE_SOURCE_DIR}/src/fv_view_describe.cpp
    # The retained scene (daz-viewer docs/retained-scene-plan.md): store, importer
    test_retained_scene.cpp
    ${CMAKE_SOURCE_DIR}/src/scene_retained.cpp
    test_retained_import.cpp
    ${CMAKE_SOURCE_DIR}/src/scene_import_gltf.cpp
    test_retained_export.cpp
    ${CMAKE_SOURCE_DIR}/src/scene_export_gltf.cpp
    # Its shader features, counted, against the detector on its export
    test_retained_features.cpp
    ${CMAKE_SOURCE_DIR}/src/scene_feature_detection.cpp
    # The figure feed: a figure straight into the retained scene, held to the import of its build and to
    # the scene's own bytes (checkFeed, header only)
    test_fv_figure_feed.cpp
    ${CMAKE_SOURCE_DIR}/src/fv_figure_feed.cpp
    # Tangents remade over the one-ring of moved vertices, held to a full recompute
    test_retained_tangents.cpp
    # Clips played and skins posed on the store, held to AnimationSystem
    test_retained_animation.cpp
    ${CMAKE_SOURCE_DIR}/src/scene_clip_player.cpp
    # Incremental equals fresh (oracle 2a): seeded random ops on the store and the feed, checked after each
    test_retained_fuzz.cpp
    # Figure Viewer texture sources outside the cooked set, Original textures (header only)
    test_fv_textures.cpp
    # OMI_environment_sky serializer: round-trip incl. unmodelled-key passthrough
    test_environment_sky.cpp
    # Phase 5c: .sky.json presets -- round trip, the sun angle OMI has no field for, strangers' keys
    test_sky_preset.cpp
    ${CMAKE_SOURCE_DIR}/src/sky_preset.cpp
    # Phase 5a: NOAA sun position, checked against astronomy rather than recorded output
    test_sun_position.cpp
    ${CMAKE_SOURCE_DIR}/src/sun_position.cpp
    # Native P4a: the Light tab's state and unit conversions, and where the sun's shadow map goes
    test_fv_lighting.cpp
    ${CMAKE_SOURCE_DIR}/src/fv_lighting.cpp
    test_shadow_fit.cpp
    # OptiX / OIDN auto-denoise schedule: interval multiples and the final image (header only)
    test_denoiser_schedule.cpp
    # Path tracer samples per frame: ending on stops, the Auto SPP controller (header only)
    test_sample_schedule.cpp
    # The tlasPolicy setting: when a TLAS sync builds rather than updates (header only)
    test_tlas_policy.cpp
    # What a selected node puts in selectedRenderNodes, whether or not the UI pass runs (header only)
    test_selection_render_nodes.cpp
    # The retained scene's GPU seam without a GPU (rs::GpuBackend): the fake's own tests
    test_fake_gpu_backend.cpp
    # What a material decides for its draws: TLAS flags and mask, the raster lists (header only)
    test_material_bits.cpp
    # The retained scene on the fake GPU, every sync held to ResidencyOracle (oracles 1 and 2b)
    test_retained_residency.cpp
    ${CMAKE_SOURCE_DIR}/src/scene_residency.cpp
    # The host's addressing on the legacy scene: uploads reach their mesh's prims, picks map back (oracle 8)
    test_fv_scene_map.cpp
    ${CMAKE_SOURCE_DIR}/src/fv_scene_map.cpp
    # Need tinygltf implementation
    ${CMAKE_SOURCE_DIR}/src/tiny_stb_implementation.cpp
    # Optional external generation bridge
    ${CMAKE_SOURCE_DIR}/src/agentic_bridge.cpp
    # Local glTF scene implementation
    ${CMAKE_SOURCE_DIR}/src/gltf_environment_sky.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_compact_model.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_compact_scene.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_material_cache.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_scene.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_scene_editor.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_image_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_scene_animation.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_scene_validator.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_version.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_scene_merger.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_scene_external_assets.cpp
    ${CMAKE_SOURCE_DIR}/src/tinygltf_converter.cpp
    ${CMAKE_SOURCE_DIR}/src/tinygltf_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_animation_pointer.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_create_tangent.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_interactivity_graph.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_interactivity_instance.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_interactivity_eval.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_interactivity_pointer.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_interactivity_scene_pointer.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_interactivity_scene_animation.cpp
    # MikkTSpace for tangent generation
    ${CMAKE_SOURCE_DIR}/third_party/MikkTSpace/mikktspace.c
    # Phase-specific tests added here as we progress
)

# Create test executable
add_executable(${TEST_NAME} ${TEST_SOURCES})

# glTF 2.1 fixtures: generated into the build tree by gltf_2_1/generate_fixtures.py, never committed.
# Without Python the folder stays empty and test_gltf_2_1.cpp skips.
set(GLTF_2_1_FIXTURES_DIR ${CMAKE_CURRENT_BINARY_DIR}/gltf_2_1)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    set(GLTF_2_1_GENERATOR ${CMAKE_CURRENT_SOURCE_DIR}/gltf_2_1/generate_fixtures.py)
    add_custom_command(
        OUTPUT ${GLTF_2_1_FIXTURES_DIR}/.generated
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${GLTF_2_1_FIXTURES_DIR}
        COMMAND ${Python3_EXECUTABLE} -I ${GLTF_2_1_GENERATOR} ${GLTF_2_1_FIXTURES_DIR}
        DEPENDS ${GLTF_2_1_GENERATOR}
        COMMENT "Generating glTF 2.1 test fixtures"
        VERBATIM)
    add_custom_target(gltf_2_1_fixtures DEPENDS ${GLTF_2_1_FIXTURES_DIR}/.generated)
    set_target_properties(gltf_2_1_fixtures PROPERTIES FOLDER "Tests")
    add_dependencies(${TEST_NAME} gltf_2_1_fixtures)
else()
    message(WARNING "Python 3 not found: glTF 2.1 test fixtures are not generated and their tests will skip.")
endif()

# Add compile definitions for tinygltf
target_compile_definitions(${TEST_NAME} PRIVATE
    GLTF_SAMPLE_ASSETS_PATH="${GLTF_SAMPLE_ASSETS_PATH}"
    # the figure library's fixtures (DAZ library, subdivision references)
    FV_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/figure/data"
    GLTF_2_1_FIXTURES_DIR="${GLTF_2_1_FIXTURES_DIR}"
)

# Add Draco support if enabled
if(USE_DRACO)
    target_include_directories(${TEST_NAME} PRIVATE
        ${draco_SOURCE_DIR}/src
        ${CMAKE_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${TEST_NAME} PRIVATE draco::draco)
    target_compile_definitions(${TEST_NAME} PRIVATE TINYGLTF_ENABLE_DRACO)
endif()

# Link with main project libraries + gtest
target_link_libraries(${TEST_NAME} PRIVATE
    GTest::gtest
    GTest::gtest_main
    fv_figure
    zlib
    # Same libraries as main project
    nvpro2::nvutils
    nvpro2::nvvk
    nvpro2::nvapp
    nvpro2::nvgui
    nvpro2::nvslang
    nvpro2::nvshaders_host
    nvpro2::nvaftermath
    nvpro2::nvgpu_monitor
    # Additional dependencies for local gltf implementation
    tinygltf
    stb
    fmt
    glm
    nvimageformats
    webp
    tinyobjloader
    meshoptimizer
)

# RapidJSON (needed for tinygltf) - tinygltf expects rapidjson headers directly
# Main project uses: ${RapidJSON_SOURCE_DIR}/rapidjson-master/include/rapidjson
set(TEST_RAPIDJSON_INCLUDE_DIR ${CMAKE_BINARY_DIR}/_deps/RapidJSON-master/rapidjson-master/include/rapidjson)

# Include directories
target_include_directories(${TEST_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/third_party/MikkTSpace
    ${TEST_RAPIDJSON_INCLUDE_DIR}
)

# Copy test resources
file(COPY ${CMAKE_SOURCE_DIR}/resources DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Register with CTest
include(GoogleTest)
gtest_discover_tests(${TEST_NAME})

# Add to build
set_target_properties(${TEST_NAME} PROPERTIES
    FOLDER "Tests"
)

# Shader code compiled as C++ (nvshaders' BSDF, the aperture sampler, the light pick, the ground, the thin scatter lobe): its own executable, since
# the BSDF header needs GLM's swizzles
set(BSDF_TEST_NAME ${PROJECT_NAME}_bsdf_tests)
add_executable(${BSDF_TEST_NAME} test_bsdf_lobes.cpp test_aperture.cpp test_light_pick.cpp test_sobol.cpp test_ground.cpp
               test_thin_scatter.cpp test_hair_bsdf.cpp)
target_link_libraries(${BSDF_TEST_NAME} PRIVATE GTest::gtest GTest::gtest_main glm)
target_include_directories(${BSDF_TEST_NAME} PRIVATE ${NVSHADERS_DIR})
gtest_discover_tests(${BSDF_TEST_NAME})
set_target_properties(${BSDF_TEST_NAME} PROPERTIES FOLDER "Tests")

#####################################################################################
# Benchmark executable (separate from tests)
set(BENCHMARK_NAME ${PROJECT_NAME}_benchmarks)

add_executable(${BENCHMARK_NAME}
    benchmark_main.cpp
    common/test_utils.cpp
    # Need tinygltf implementation
    ${CMAKE_SOURCE_DIR}/src/tiny_stb_implementation.cpp
    # Local glTF scene implementation
    ${CMAKE_SOURCE_DIR}/src/gltf_animation_pointer.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_environment_sky.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_compact_model.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_compact_scene.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_scene.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_scene_editor.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_image_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_scene_animation.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_scene_validator.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_version.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_scene_merger.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_scene_external_assets.cpp
    ${CMAKE_SOURCE_DIR}/src/tinygltf_converter.cpp
    ${CMAKE_SOURCE_DIR}/src/tinygltf_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_create_tangent.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_interactivity_graph.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_interactivity_instance.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_interactivity_eval.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_interactivity_pointer.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_interactivity_scene_pointer.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_interactivity_scene_animation.cpp
    # The retained scene (daz-viewer docs/retained-scene-plan.md): store, importer, figure feed
    ${CMAKE_SOURCE_DIR}/src/scene_retained.cpp
    ${CMAKE_SOURCE_DIR}/src/scene_import_gltf.cpp
    ${CMAKE_SOURCE_DIR}/src/scene_export_gltf.cpp
    ${CMAKE_SOURCE_DIR}/src/scene_clip_player.cpp
    ${CMAKE_SOURCE_DIR}/src/scene_feature_detection.cpp
    ${CMAKE_SOURCE_DIR}/src/fv_figure_feed.cpp
    # MikkTSpace for tangent generation
    ${CMAKE_SOURCE_DIR}/third_party/MikkTSpace/mikktspace.c
)

# the Box Figure the retained-scene benchmarks feed
target_compile_definitions(${BENCHMARK_NAME} PRIVATE FV_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/figure/data")

# Add Draco support if enabled
if(USE_DRACO)
    target_include_directories(${BENCHMARK_NAME} PRIVATE
        ${draco_SOURCE_DIR}/src
        ${CMAKE_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${BENCHMARK_NAME} PRIVATE draco::draco)
    target_compile_definitions(${BENCHMARK_NAME} PRIVATE TINYGLTF_ENABLE_DRACO)
endif()

target_link_libraries(${BENCHMARK_NAME} PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    # Same libraries as test executable
    fv_figure
    zlib
    nvpro2::nvutils
    nvpro2::nvvk
    nvpro2::nvapp
    nvpro2::nvgui
    nvpro2::nvslang
    nvpro2::nvshaders_host
    nvpro2::nvaftermath
    nvpro2::nvgpu_monitor
    # Additional dependencies for local gltf implementation
    tinygltf
    stb
    fmt
    glm
    nvimageformats
    webp
    tinyobjloader
    meshoptimizer
)

target_include_directories(${BENCHMARK_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/third_party/MikkTSpace
    ${TEST_RAPIDJSON_INCLUDE_DIR}
)

set_target_properties(${BENCHMARK_NAME} PROPERTIES
    FOLDER "Tests"
)
