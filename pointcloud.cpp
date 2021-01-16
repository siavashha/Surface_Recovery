#include "pointcloud.h"

PointCloud::PointCloud()
{
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloud::CalcKeyPoints(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud  )
{
    pcl::PointCloud<int> sampled_indices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_keypoints
            (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::UniformSampling<pcl::PointXYZ> uniform_sampling;
    uniform_sampling.setInputCloud (cloud);
    uniform_sampling.setRadiusSearch (0.1);
    uniform_sampling.compute (sampled_indices);
    pcl::copyPointCloud (*cloud, sampled_indices.points, *cloud_keypoints);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud_keypoints , *cloud_keypoints , indices);
    return cloud_keypoints;
}

pcl::ModelCoefficients::Ptr PointCloud::Ground_Object_disscrimination(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud ,
        pcl::PointCloud<pcl::PointXYZ>::Ptr &grourd_cloud ,
        pcl::PointCloud<pcl::PointXYZ>::Ptr &nonGround_cloud  )
{
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.20);
    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setNegative (false);
    extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    extract.filter (*grourd_cloud);
    extract.setNegative (true);
    extract.filter (*nonGround_cloud);
    return coefficients;
}

std::vector < pcl::PointCloud<pcl::PointXYZ>::Ptr > PointCloud::Clustering(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree
            (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.2); // 2cm
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);
    int j = 0;
    std::vector < pcl::PointCloud<pcl::PointXYZ>::Ptr > clusters;
    for (std::vector<pcl::PointIndices>::const_iterator it =
         cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster
                (new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit =
             it->indices.begin (); pit != it->indices.end (); pit++)
            cloud_cluster->points.push_back (cloud->points[*pit]); //*
        clusters.push_back(cloud_cluster);
    }
    return clusters;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloud::MLSSurface
    (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud
            (new pcl::PointCloud<pcl::PointXYZ>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr
            tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    std::cout << " MLS  inputsize= " << cloud->size() << std::endl;
    pcl::MovingLeastSquares<pcl::PointXYZ , pcl::PointXYZ> mls;
    mls.setInputCloud(cloud);
    mls.setSearchRadius(0.5);
    mls.setPolynomialFit(true);
    mls.setPolynomialOrder(2.0);
    mls.setUpsamplingMethod(
    pcl::MovingLeastSquares<pcl::PointXYZ,pcl::PointXYZ>::SAMPLE_LOCAL_PLANE);
    mls.setUpsamplingRadius(0.15);
    mls.setUpsamplingStepSize(0.01);
    mls.setSearchMethod(tree);
    mls.process(*out_cloud);
    std::cout << " after MLS size= " << out_cloud->size() << std::endl;
    return out_cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloud::ColorCloud
(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, cv::Mat image, cv::Mat Tr)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgbCloud
            (new pcl::PointCloud<pcl::PointXYZRGB>());

    for (int32_t i = 0; i < cloud->size(); i++)
    {
        pcl::PointXYZ p;
        p = cloud->points[i];
        cv::Mat pMat(4,1,CV_64F,cv::Scalar(0));
        pMat.at<double>(0,0) = p.x;
        pMat.at<double>(1,0) = p.y;
        pMat.at<double>(2,0) = p.z;
        pMat.at<double>(3,0) = 1.;
        cv::Mat qMat(4,1,CV_64F,cv::Scalar(0.));
        qMat = Tr * pMat;
        int qx = qMat.at<double>(0,0)/qMat.at<double>(2,0);
        int qy = qMat.at<double>(1,0)/qMat.at<double>(2,0);
        if ((qx < 0) || (qy < 0) || (qx > image.cols) || (qy > image.rows))
            continue;
        cv::Vec3b intensity = image.at< cv::Vec3b >(qy , qx);
        uchar blue = intensity.val[0];
        uchar green = intensity.val[1];
        uchar red = intensity.val[2];
        pcl::PointXYZRGB q;
        q.x = p.x;
        q.y = p.y;
        q.z = p.z;
        uint32_t rgb = (static_cast<uint32_t>(red) << 16 |
                        static_cast<uint32_t>(green) << 8 | static_cast<uint32_t>(blue));
        q.rgb = *reinterpret_cast<float*>(&rgb);
        rgbCloud->points.push_back(q);
    }
    return rgbCloud;
}

