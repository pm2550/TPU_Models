# Makefile for Native C++ TPU Benchmark
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -fPIC
INCLUDES = -I./tensorflowlite_c_2.5.0_arm64/include -I/usr/include
LIBS = -L./tensorflowlite_c_2.5.0_arm64 -L/usr/lib/aarch64-linux-gnu -ltensorflowlite_c -ledgetpu -lpthread -ldl
TARGET = plot_native
SOURCE = plot_native.cpp

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCE) $(LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Run the program
run: $(TARGET)
	LD_LIBRARY_PATH=./tensorflowlite_c_2.5.0_arm64:$$LD_LIBRARY_PATH ./$(TARGET)

.PHONY: all clean run
