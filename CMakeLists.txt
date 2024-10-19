cmake_minimum_required(VERSION 3.5)
project(pointcloud_to_gridmap_cuda)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

find_package(CUDAToolkit REQUIRED)

include_directories(include)

set(CUDA_SOURCES
    src/pointcloud_to_gridmap_cuda.cu
)

add_executable(pointcloud_to_gridmap_cuda ${CUDA_SOURCES})
ament_target_dependencies(pointcloud_to_gridmap_cuda
  rclcpp
  sensor_msgs
  nav_msgs
)

target_link_libraries(pointcloud_to_gridmap_cuda ${CUDAToolkit_LIBRARIES})

install(TARGETS
  pointcloud_to_gridmap_cuda
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
