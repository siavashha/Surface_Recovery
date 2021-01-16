#ifndef READINGLABELS_H
#define READINGLABELS_H
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <map>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/crop_box.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <pcl/surface/bilateral_upsampling.h>
#include <pcl/surface/impl/bilateral_upsampling.hpp>
#include <pcl/surface/mls.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

//#include <sstream>
#include "Label.h"


class ReadingLabels
{
private:
    std::string m_TRN_VEL_DIR;
    std::string m_TRN_LCAM_DIR;
    std::string m_TRN_RCAM_DIR;
    std::string m_TRN_LBL_DIR;
    std::string m_TRN_CALIB_DIR;
public:
    ReadingLabels();
    ReadingLabels(std::string TRN_VEL_DIR, std::string TRN_LCAM_DIR,
    std::string TRN_RCAM_DIR,std::string TRN_LBL_DIR,std::string TRN_CALIB_DIR);
    int findEpochs(std::string TRN_VEL_DIR ,int i);
    std::vector<Label> ReadLabels(std::string filename, int epoch);
    std::string FindEpochFileName(int epochNo , std::string  TRN_DIR );
    std::string FindDatasetFileName(int datasetNo , std::string  TRN_DIR );
    std::vector<object_List> ReadData
        (std::string lbl_dir, std::string ls_dir , std::string img_dir , std::string calib_dir);
    Calib ReadCalibs(std::string calib_dir );
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ColorPointCloud
//    (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, cv::Mat image, cv::Mat Tr);
    std::vector<std::string> Read_Lbl_Calib(int datasetNo);
    std::vector<std::string> Read_LS_Img(int datasetNo, int EpochNo);
    void ReadPCD(std::string filename , pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, bool filtered , cv::Mat Tr_velo_to_cam);
    void ReadPCD(std::string filename , pcl::PointCloud<pcl::PointXYZ>::Ptr cloud , cv::Mat Tr_velo_to_cam);

    std::vector < std::vector<Label> > ReadLabelMap(std::string filename,int numOfEpochs);
    //void Upsample();
    pcl::PointCloud<pcl::PointXYZ>::Ptr Upsample
    (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
};

#endif // READINGLABELS_H
