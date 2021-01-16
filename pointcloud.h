#ifndef POINTCLOUD_H
#define POINTCLOUD_H
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/boundary.h>
#include <pcl/surface/mls.h>
#include <boost/lexical_cast.hpp>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

class PointCloud
{
public:
    PointCloud();
    pcl::PointCloud<pcl::PointXYZ>::Ptr CalcKeyPoints(
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud  );
    // ground removal
    pcl::ModelCoefficients::Ptr Ground_Object_disscrimination(
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud ,
            pcl::PointCloud<pcl::PointXYZ>::Ptr &grourd_cloud ,
            pcl::PointCloud<pcl::PointXYZ>::Ptr &nonGround_cloud  );
    // Clustering
    std::vector < pcl::PointCloud<pcl::PointXYZ>::Ptr >
    Clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr MLSSurface
        (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ColorCloud
    (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, cv::Mat image, cv::Mat Tr);
};

#endif // POINTCLOUD_H
