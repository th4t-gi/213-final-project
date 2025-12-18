CXX := clang++
CUXX := nvcc
CXXFLAGS := -std=c++20
CUXXFLAGS := 
INCLUDES_FLAGS := -lsqlite3

EXEC := main
DEPS := utils kernel worker

# Directories
SRC_DIR := src
BUILD_DIR := build

# Source and Object Files
OBJ_DEPS := $(addprefix $(BUILD_DIR)/, $(addsuffix .o, $(DEPS)))
EXEC_BINS := build/main
# EXEC_BINS := $(addprefix $(BUILD_DIR)/, $(EXEC))
# # Headers
HDRS := $(wildcard $(SRC_DIR)/*.h)

.PHONY: all fresh clean

all: $(EXEC_BINS)

fresh:
	make clean && make all

build:
	mkdir -p $(BUILD_DIR)


$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(HDRS) | build
	$(CUXX) $(CUXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HDRS) | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/main: $(SRC_DIR)/main.cu $(OBJ_DEPS) | build
	$(CUXX) $(CUXXFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD_DIR)

