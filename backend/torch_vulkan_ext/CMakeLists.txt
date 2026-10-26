cmake_minimum_required(VERSION 3.18)
project(tvkvk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Vulkan REQUIRED)

# Standalone raw-Vulkan backend. Deliberately links against libvulkan only:
# no Kompute (its eDevice tensors allocate a full host staging mirror, which
# would double VRAM for resident weights) and no Python (the Python layer loads
# this through ctypes, so no Python.h / pybind11 is needed).
add_library(tvkvk SHARED
  csrc/vk_backend.cpp
  csrc/vk_cabi.cpp
)
target_include_directories(tvkvk PRIVATE ${Vulkan_INCLUDE_DIRS} csrc/)
target_link_libraries(tvkvk PRIVATE ${Vulkan_LIBRARIES})

set_target_properties(tvkvk PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
  PREFIX ""
)
