cmake_minimum_required(VERSION 3.25)
project(riecoin_network_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
get_filename_component(RIECOIN_COMPILER_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(RIECOIN_UCRT_ROOT "${RIECOIN_COMPILER_BIN}" DIRECTORY)
set(RIECOIN_UCRT_INCLUDE "${RIECOIN_UCRT_ROOT}/include" CACHE PATH
    "UCRT64 include directory containing Boost.JSON and Boost.Multiprecision")
set(RIECOIN_UCRT_LIBRARY "${RIECOIN_UCRT_ROOT}/lib" CACHE PATH
    "UCRT64 library directory")

if(NOT EXISTS "${RIECOIN_UCRT_INCLUDE}/boost/json.hpp")
  message(FATAL_ERROR "Boost.JSON not found under RIECOIN_UCRT_INCLUDE")
endif()

if(EXISTS "${RIECOIN_UCRT_LIBRARY}/libboost_json-mt.a")
  set(RIECOIN_BOOST_JSON "${RIECOIN_UCRT_LIBRARY}/libboost_json-mt.a")
else()
  find_library(RIECOIN_BOOST_JSON
    NAMES boost_json-mt boost_json
    PATHS "${RIECOIN_UCRT_LIBRARY}"
    NO_DEFAULT_PATH
    REQUIRED)
endif()
if(NOT EXISTS "${RIECOIN_UCRT_LIBRARY}/libgmp.a")
  message(FATAL_ERROR "Pinned static GMP library not found")
endif()

else()
  # Avoid a distro-specific libboost_json.so.1.xx runtime dependency.
  # The header/library version selected by this source build stays matched.
  set(Boost_USE_STATIC_LIBS ON)
  find_package(Boost 1.75 REQUIRED COMPONENTS json)
  find_package(OpenSSL REQUIRED COMPONENTS Crypto)
  find_path(GMP_INCLUDE_DIR gmp.h REQUIRED)
  find_library(GMP_LIBRARY gmp REQUIRED)
endif()

add_executable(riecoin-network-backend
  riecoin_network_backend.cpp
  riecoin_stratum_protocol.cpp
  riecoin_stage_bridge.cpp
)
if(WIN32)
target_include_directories(riecoin-network-backend PRIVATE "${RIECOIN_UCRT_INCLUDE}")
target_compile_definitions(riecoin-network-backend PRIVATE
  WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00 BOOST_JSON_STATIC_LINK)
else()
target_include_directories(riecoin-network-backend PRIVATE "${GMP_INCLUDE_DIR}")
target_link_libraries(riecoin-network-backend PRIVATE Boost::json OpenSSL::Crypto "${GMP_LIBRARY}")
endif()
target_compile_options(riecoin-network-backend PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic -Wconversion -Wshadow>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /EHsc>)
if(WIN32)
target_link_libraries(riecoin-network-backend PRIVATE
  "${RIECOIN_BOOST_JSON}" "${RIECOIN_UCRT_LIBRARY}/libgmp.a" ws2_32 bcrypt)
target_link_options(riecoin-network-backend PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-static -static-libgcc -static-libstdc++>)
endif()
set_target_properties(riecoin-network-backend PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
install(TARGETS riecoin-network-backend RUNTIME DESTINATION bin)
