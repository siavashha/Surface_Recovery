#ifndef PROPOSED_H
#define PROPOSED_H
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/morphological_filter.h>
#include <pcl/filters/median_filter.h>
#include <opencv2/opencv.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <boost/lexical_cast.hpp>

class Proposed
{
    cv::Mat m_lImage;
    cv::Mat m_Tr;
public:
    Proposed();
    void Reconstruct(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud );
    void ImageAdding(cv::Mat lImage , cv::Mat Tr);

};

#endif // PROPOSED_H
