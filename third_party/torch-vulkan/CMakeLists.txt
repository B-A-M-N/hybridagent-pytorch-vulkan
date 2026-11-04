cmake_minimum_required(VERSION 3.18)
project(torch_vulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python (needed for torch/extension.h -> Python.h)
find_package(Python3 COMPONENTS Development REQUIRED)

# Find PyTorch
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)

# Find Vulkan SDK
find_package(Vulkan REQUIRED)

# Find Kompute -- built locally by scripts/install.sh and installed under
# third_party/kompute_install (v0.9.0). We no longer FetchContent it, so the
# build is offline-reproducible and pinned to the audited revision.
set(KOMPUTE_PREFIX "${CMAKE_SOURCE_DIR}/../kompute_install")
find_path(KOMPUTE_INCLUDE_DIR kompute/Kompute.hpp HINTS ${KOMPUTE_PREFIX}/include REQUIRED)
find_library(KOMPUTE_LIBRARY kompute HINTS ${KOMPUTE_PREFIX}/lib REQUIRED)
# Kompute splits its logger into a separate archive (libkp_logger.a) and
# libkompute.a leaves `logger::setupLogger()` undefined. Link it with
# --whole-archive so the symbol is actually pulled in.
find_library(KOMPUTE_LOGGER_LIBRARY kp_logger HINTS ${KOMPUTE_PREFIX}/lib REQUIRED)
find_library(KOMPUTE_FMT_LIBRARY fmt HINTS ${KOMPUTE_PREFIX}/lib REQUIRED)
# --whole-archive is required: these static archives reference each other's
# symbols through internal helpers that the linker would otherwise drop.
set(KOMPUTE_LIBRARIES
  -Wl,--whole-archive ${KOMPUTE_LIBRARY} ${KOMPUTE_LOGGER_LIBRARY} ${KOMPUTE_FMT_LIBRARY} -Wl,--no-whole-archive)
message(STATUS "Kompute: include=${KOMPUTE_INCLUDE_DIR} lib=${KOMPUTE_LIBRARY} logger=${KOMPUTE_LOGGER_LIBRARY} fmt=${KOMPUTE_FMT_LIBRARY}")

# ---- torch-vulkan (patched) -----------------------------------------------
# The only patch is in csrc/vulkan_allocator.cpp: after creating a Kompute
# tensor we register data_ptr -> VkBuffer in tvk::Registry, so the raw-Vulkan
# extension can bind the tensor's device buffer directly and keep weights
# resident in VRAM without a host mirror.
add_library(torch_vulkan_C SHARED
  csrc/torch_vulkan.cpp
  csrc/vulkan_context.cpp
  csrc/vulkan_allocator.cpp
  csrc/ops/mm.cpp
  csrc/vulkan_engine.cpp
  ${CMAKE_SOURCE_DIR}/../../backend/torch_vulkan_ext/csrc/tvk_registry.cpp
)

target_include_directories(torch_vulkan_C PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${Vulkan_INCLUDE_DIRS}
  ${KOMPUTE_INCLUDE_DIR}
  ${Python3_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIRS}
  csrc/
  ${CMAKE_SOURCE_DIR}/../../backend/torch_vulkan_ext/csrc/
)

target_link_libraries(torch_vulkan_C PRIVATE Python3::Python)
target_link_libraries(torch_vulkan_C PRIVATE
  ${TORCH_LIBRARIES}
  ${Vulkan_LIBRARIES}
  ${KOMPUTE_LIBRARIES}
)

# Output as _C.so for Python import
set_target_properties(torch_vulkan_C PROPERTIES
  OUTPUT_NAME "_C"
  PREFIX ""
  SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/torch_vulkan"
)

# ---- torch_vulkan_ext -----------------------------------------------------
# The raw-Vulkan extension: rmsnorm, rope, silu_gate, kv_store, attention,
# argmax, slice/gather, gelu_erf, broadcast_add.
add_library(torch_vulkan_ext_C SHARED
  ${CMAKE_SOURCE_DIR}/../../backend/torch_vulkan_ext/csrc/vk_backend.cpp
  ${CMAKE_SOURCE_DIR}/../../backend/torch_vulkan_ext/csrc/vk_ops.cpp
  ${CMAKE_SOURCE_DIR}/../../backend/torch_vulkan_ext/csrc/tvk_registry.cpp
)

target_include_directories(torch_vulkan_ext_C PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${Vulkan_INCLUDE_DIRS}
  ${KOMPUTE_INCLUDE_DIR}
  ${Python3_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIRS}
  csrc/
  ${CMAKE_SOURCE_DIR}/../../backend/torch_vulkan_ext/csrc/
)

target_link_libraries(torch_vulkan_ext_C PRIVATE Python3::Python)
target_link_libraries(torch_vulkan_ext_C PRIVATE
  ${TORCH_LIBRARIES}
  ${Vulkan_LIBRARIES}
  ${KOMPUTE_LIBRARIES}
)

set_target_properties(torch_vulkan_ext_C PROPERTIES
  OUTPUT_NAME "_C"
  PREFIX ""
  SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../../backend/torch_vulkan_ext/torch_vulkan_ext"
)
