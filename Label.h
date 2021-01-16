#ifndef LABEL_H
#define LABEL_H
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <pcl/point_types.h>
#include <Eigen/Eigen>
#include <Eigen/Core>

struct Label{
    int frame; //       Frame within the sequence where the object appearers
    int track_id;//     Unique tracking id of this object within this sequence
    std::string object_type;//         Describes the type of object: 'Car', 'Van', 'Truck',
             //           'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
             //           'Misc' or 'DontCare'
    bool truncated;//    Float from 0 (non-truncated) to 1 (truncated), where
                   //     truncated refers to the object leaving image boundaries
    int occluded;//     Integer (0,1,2,3) indicating occlusion state:
                 //       0 = fully visible, 1 = partly occluded
                 //       2 = largely occluded, 3 = unknown
    double alpha; //       Observation angle of object, ranging [-pi..pi]
    Eigen::Vector4f  bbox;//         2D bounding box of object in the image (0-based index):
                        //contains left, top, right, bottom pixel coordinates
    Eigen::Vector3f dimensions;//   3D object dimensions: height, width, length (in meters)
    Eigen::Vector3f location;//     3D object location x,y,z in camera coordinates (in meters)
    double rotation_y; //   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    float score;//        Only for results: Float, indicating confidence in
                 //       detection, needed for p/r curves, higher is better.
};

struct Calib{
    cv::Mat P0;// 4*4
    cv::Mat P1;// 4*4
    cv::Mat P2;// 4*4
    cv::Mat P3;// 4*4
    cv::Mat R_rect;// 3*3
    cv::Mat Tr_velo_cam;// 4*4
    cv::Mat Tr_imu_velo;// 4*4
    cv::Mat Tr_P2;
    cv::Mat Tr_P3;
};


struct object_List{
    std::vector<Label> labels;
    Calib calibration;
    std::string ls_filename;
    std::string img_filename;
    std::string lbl_filename;
};

#endif // LABEL_H
