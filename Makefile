CXX := clang++
CUXX := nvcc
CXXFLAGS := -std=c++20 -O3
CUXXFLAGS := 
INCLUDES_FLAGS := -lsqlite3

EXEC := main
CUDEPS := kernel
DEPS := utils

# Directories
SRC_DIR := src
BUILD_DIR := build

# Source and Object Files
OBJ_DEPS := $(addprefix $(BUILD_DIR)/, $(addsuffix .o, $(DEPS)))
EXEC_BINS := build/main
# EXEC_BINS := $(addprefix $(BUILD_DIR)/, $(EXEC))
# # Headers
# HDRS := $(wildcard $(SRC_DIR)/*.h)

.PHONY: all fresh clean

all: $(EXEC_BINS)

fresh:
	make clean && make all

build:
	mkdir -p $(BUILD_DIR)


$(BUILD_DIR)/kernel.o: $(SRC_DIR)/kernel.cu $(SRC_DIR)/kernel.h | build
	$(CUXX) $(CUXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HDRS) | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/main: $(SRC_DIR)/main.cpp $(OBJ_DEPS) | build
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD_DIR)

