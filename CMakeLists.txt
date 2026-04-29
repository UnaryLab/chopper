cmake_minimum_required(VERSION 3.22)
project(chopper VERSION 0.10.0 LANGUAGES CXX)

enable_language(HIP)

add_subdirectory(chopper/profile/telemetry/src)
