#include <iostream>
#include <vector>
#include <CL/cl.h>

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error during " << operation << ": " << err << std::endl;
        exit(1);
    }
}

int main() {
    cl_uint numPlatforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "No OpenCL platforms found or failed to query platforms. Error: " << err << std::endl;
        return 1;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    checkError(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr), "clGetPlatformIDs");

    std::cout << "Found " << numPlatforms << " OpenCL platform(s):" << std::endl;

    for (cl_uint i = 0; i < numPlatforms; ++i) {
        char platformName[256];
        char platformVendor[256];
        char platformVersion[256];

        checkError(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr), "CL_PLATFORM_NAME");
        checkError(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(platformVendor), platformVendor, nullptr), "CL_PLATFORM_VENDOR");
        checkError(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(platformVersion), platformVersion, nullptr), "CL_PLATFORM_VERSION");

        std::cout << "\nPlatform " << i << ":" << std::endl;
        std::cout << "  Name:    " << platformName << std::endl;
        std::cout << "  Vendor:  " << platformVendor << std::endl;
        std::cout << "  Version: " << platformVersion << std::endl;

        cl_uint numDevices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        if (err != CL_SUCCESS || numDevices == 0) {
            std::cout << "  No devices found for this platform." << std::endl;
            continue;
        }

        std::vector<cl_device_id> devices(numDevices);
        checkError(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr), "clGetDeviceIDs");

        std::cout << "  Found " << numDevices << " device(s):" << std::endl;

        for (cl_uint j = 0; j < numDevices; ++j) {
            char deviceName[256];
            char deviceVersion[256];
            char driverVersion[256];
            cl_device_type deviceType;
            cl_ulong globalMemSize;
            cl_ulong localMemSize;
            cl_ulong maxAllocSize;
            size_t maxWorkGroupSize;
            cl_uint maxComputeUnits;
            cl_uint maxClockFreq;

            checkError(clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr), "CL_DEVICE_NAME");
            checkError(clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(deviceVersion), deviceVersion, nullptr), "CL_DEVICE_VERSION");
            checkError(clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(driverVersion), driverVersion, nullptr), "CL_DRIVER_VERSION");
            checkError(clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr), "CL_DEVICE_TYPE");
            checkError(clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, nullptr), "CL_DEVICE_GLOBAL_MEM_SIZE");
            checkError(clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, nullptr), "CL_DEVICE_LOCAL_MEM_SIZE");
            checkError(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAllocSize), &maxAllocSize, nullptr), "CL_DEVICE_MAX_MEM_ALLOC_SIZE");
            checkError(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr), "CL_DEVICE_MAX_WORK_GROUP_SIZE");
            checkError(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, nullptr), "CL_DEVICE_MAX_COMPUTE_UNITS");
            checkError(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFreq), &maxClockFreq, nullptr), "CL_DEVICE_MAX_CLOCK_FREQUENCY");

            std::cout << "\n    Device " << j << ":" << std::endl;
            std::cout << "      Name:                  " << deviceName << std::endl;
            std::cout << "      Type:                  " 
                      << ((deviceType & CL_DEVICE_TYPE_GPU) ? "GPU " : "")
                      << ((deviceType & CL_DEVICE_TYPE_CPU) ? "CPU " : "")
                      << ((deviceType & CL_DEVICE_TYPE_ACCELERATOR) ? "ACCELERATOR " : "")
                      << std::endl;
            std::cout << "      OpenCL C Version:      " << deviceVersion << std::endl;
            std::cout << "      Driver Version:        " << driverVersion << std::endl;
            std::cout << "      Compute Units:         " << maxComputeUnits << std::endl;
            std::cout << "      Max Clock Frequency:   " << maxClockFreq << " MHz" << std::endl;
            std::cout << "      Global Memory Size:    " << (globalMemSize / (1024 * 1024)) << " MB" << std::endl;
            std::cout << "      Max Alloc Memory Size: " << (maxAllocSize / (1024 * 1024)) << " MB" << std::endl;
            std::cout << "      Local Memory Size:     " << (localMemSize / 1024) << " KB" << std::endl;
            std::cout << "      Max Workgroup Size:    " << maxWorkGroupSize << std::endl;
        }
    }
    return 0;
}
