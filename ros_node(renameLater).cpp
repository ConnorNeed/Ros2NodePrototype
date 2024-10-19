#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "grid_map_msgs/msg/grid_map.hpp"
#include "grid_map_core/GridMap.hpp"

// TODO: Seperate into header and source file

class CudaPointCloudToGridMapNode : public rclcpp::Node {
public:
    CudaPointCloudToGridMapNode() : Node("cuda_pointcloud_to_gridmap") {
        // TODO Make topics configurable

        // TODO make resolution configurable 

        pointcloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/points", 10,
            std::bind(&CudaPointCloudToGridMapNode::pointcloud_callback, this, std::placeholders::_1));

        gridmap_publisher_ = this->create_publisher<grid_map_msgs::msg::GridMap>("output_gridmap", 10);
    }

private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

        // Prepare point data for GPU
        // TODO optimize copies
        pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
        pcl::fromROSMsg(*msg, pcl_cloud);
        int num_points = pcl_cloud.size();
        if (num_points == 0) {
            RCLCPP_WARN(this->get_logger(), "Received empty point cloud.");
            return;
        }
        std::vector<float> points(num_points * 3);
        for (size_t i = 0; i < num_points; ++i) {
            points[i * 3] = pcl_cloud.points[i].x;
            points[i * 3 + 1] = pcl_cloud.points[i].y;
            points[i * 3 + 2] = pcl_cloud.points[i].z;
        }

        // Allocate GPU Memory
        float *d_points;
        float *d_grid_map;
        size_t grid_map_size = width * height * sizeof(float);
        cudaMalloc(&d_points, num_points * sizeof(float) * 3);
        cudaMalloc(&d_grid_map, grid_map_size);
        
        // Copy Data to GPU
        cudaMemcpy(d_points, points.data(), points.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Launch CUDA Kernel
        int blockSize = 256;
        int numBlocks = (num_points + blockSize - 1) / blockSize;
        convertPointCloudToGridMapKernel<<<numBlocks, blockSize>>>(d_points, num_points, d_grid_map, width, height, resolution);
        
        // Copy Results Back to CPU
        std::vector<float> grid_map_data(width * height);
        cudaMemcpy(grid_map_data.data(), d_grid_map, grid_map_size, cudaMemcpyDeviceToHost);
        
        // Create and Publish GridMap Message
        grid_map::GridMap grid_map;
        grid_map.setFrameId("map");
        grid_map.setGeometry(grid_map::Length(width * resolution, height * resolution), resolution);
        grid_map.add("elevation", grid_map_data);
        
        // Convert to msg type
        grid_map_msgs::msg::GridMap grid_map_msg;
        grid_map::GridMapRosConverter::toMessage(grid_map, grid_map_msg);
        
        gridmap_publisher_->publish(grid_map_msg);

        cudaFree(d_points);
        cudaFree(d_grid_map);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscriber_;
    rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr gridmap_publisher_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CudaPointCloudToGridMapNode>());
    rclcpp::shutdown();
    return 0;
}
