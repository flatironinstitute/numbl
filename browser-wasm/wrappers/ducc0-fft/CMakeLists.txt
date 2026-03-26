cmake_minimum_required(VERSION 3.16)

project(numbl_ducc0_fft_wrapper LANGUAGES CXX)

if(NOT DEFINED DUCC0_SRC_ROOT OR DUCC0_SRC_ROOT STREQUAL "")
  message(FATAL_ERROR "Set DUCC0_SRC_ROOT to the ducc checkout root")
endif()

add_library(numbl_ducc0_fft_wrapper OBJECT ducc0_fft_wrapper.cpp)

target_include_directories(numbl_ducc0_fft_wrapper PRIVATE
  "${DUCC0_SRC_ROOT}/src"
)

target_compile_features(numbl_ducc0_fft_wrapper PRIVATE cxx_std_17)

target_compile_definitions(numbl_ducc0_fft_wrapper PRIVATE
  DUCC0_WRAPPER_BUILD=1
)
