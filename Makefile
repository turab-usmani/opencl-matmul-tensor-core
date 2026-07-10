# Makefile for OpenCL Custom Tensor Core Project

CXX = g++
CXXFLAGS = -O3 -Wall -Ithird_party/OpenCL-Headers
LDFLAGS = C:\Windows\System32\OpenCL.dll

TARGET = opencl_matmul.exe
QUERY_TARGET = query_device.exe

all: $(TARGET) $(QUERY_TARGET)

$(TARGET): src/main.cpp
	$(CXX) $(CXXFLAGS) src/main.cpp -o $(TARGET) $(LDFLAGS)

$(QUERY_TARGET): src/query_device.cpp
	$(CXX) $(CXXFLAGS) src/query_device.cpp -o $(QUERY_TARGET) $(LDFLAGS)

clean:
	del /q $(TARGET) $(QUERY_TARGET) 2>nul || rm -f $(TARGET) $(QUERY_TARGET)
