find_package(rocprofiler-sdk REQUIRED HINTS ${ROCM_PATH} /opt/rocm)

add_library(chopper_device_counters SHARED device_counters_tool.cpp)

set_target_properties(chopper_device_counters PROPERTIES
    OUTPUT_NAME "chopper_device_counters"
    PREFIX "lib"
    CXX_STANDARD 17
    HIP_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
)

target_link_libraries(chopper_device_counters PRIVATE
    rocprofiler-sdk::rocprofiler-sdk
)

target_compile_options(chopper_device_counters PRIVATE
    -Wall -Wextra
    $<$<CONFIG:Debug>:-O0 -ggdb>
    $<$<CONFIG:Release>:-O2>
)

install(TARGETS chopper_device_counters
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
