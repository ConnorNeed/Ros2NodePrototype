#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "pcl/conversions.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include <vector>

__device__ void atomicMax(float* address, float value) {
    float old = *address;
    while (value > old) {
        if (atomicCAS(address, old, value) == old) {
            break;
        }
        old = *address;
    }
}

__global__ void convertPointCloudToGridMapKernel(const float* points, int num_points, float* grid_map, int width, int height, float resolution) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        int grid_x = static_cast<int>(points[idx * 3] / resolution);
        int grid_y = static_cast<int>(points[idx * 3 + 1] / resolution);

        if (grid_x >= 0 && grid_x < width && grid_y >= 0 && grid_y < height) {
            atomicMax(&grid_map[grid_y * width + grid_x], points[idx * 3 + 2]);
        }
    }
}