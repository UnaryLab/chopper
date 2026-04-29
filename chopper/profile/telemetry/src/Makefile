ROCM_PATH ?= /opt/rocm
HIPCXX    ?= $(ROCM_PATH)/bin/hipcc

CXXFLAGS  := -std=c++17 -Wall -Wextra -O2
CPPFLAGS  := -D__HIP_PLATFORM_AMD__ -isystem $(ROCM_PATH)/include
LDFLAGS   := -L$(ROCM_PATH)/lib
LIBS      := -lrocprofiler-sdk

LIB_DIR   := ../lib
TARGET    := $(LIB_DIR)/libchopper_device_counters.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): device_counters_tool.cpp | $(LIB_DIR)
	$(HIPCXX) $(CXXFLAGS) $(CPPFLAGS) -fPIC -shared -o $@ $< $(LDFLAGS) $(LIBS)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

clean:
	$(RM) $(TARGET)
