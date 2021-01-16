#include <QCoreApplication>

#include "Readlabels.h"
#include "pointcloud.h"
#include "proposed.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    //data sources
    std::string  TRN_VEL_DIR = "/home/sia/Documents/Data/tracking_Dataset/Dataset/training/velodyne";
    std::string  TRN_LCAM_DIR = "/home/sia/Documents/Data/tracking_Dataset/Dataset/training/image_02";
    std::string  TRN_RCAM_DIR = "/home/sia/Documents/Data/tracking_Dataset/Dataset/training/image_03";
    std::string  TRN_LBL_DIR = "/home/sia/Documents/Data/tracking_Dataset/Dataset/training/label_02";
    std::string  TRN_CALIB_DIR = "/home/sia/Documents/Data/tracking_Dataset/Dataset/training/calib";
    ReadingLabels* readLabel =  new ReadingLabels(TRN_VEL_DIR,TRN_LCAM_DIR, TRN_RCAM_DIR, TRN_LBL_DIR,TRN_CALIB_DIR);
    PointCloud* pcd_proc = new PointCloud();
    Proposed* constructor = new Proposed();
    std::cout << " poinst: " << 0 << std::endl;

    // Iterator on datasets
    for (int i = 0; i < 21; i++)
    {
        int numOfEpochs = readLabel->findEpochs(TRN_VEL_DIR , i);
        std::vector<std::string> files = readLabel->Read_Lbl_Calib(i);
        std::string lbl_filename = files[0];
        std::string calib_filename = files[1];
        std::cout << i << ":" << numOfEpochs << std::endl;
        Calib calibration = readLabel->ReadCalibs(calib_filename);
        // Iterator on epochs
        for (int j = 0; j < numOfEpochs; j++)
        {
            std::vector<Label> label_list =
                    readLabel->ReadLabels(lbl_filename , j);
            std::vector<std::string> files = readLabel->Read_LS_Img(i,j);
            std::string ls_filename = files[0];
            std::string imgL_filename = files[1];
            std::string imgR_filename = files[2];
            cv::Mat lImage = cv::imread(imgL_filename ,cv::IMREAD_COLOR);
            cv::Mat rImage = cv::imread(imgR_filename ,cv::IMREAD_COLOR);
            cv::Mat filtered_image;
            cv::bilateralFilter(lImage,filtered_image,23,25,5);
            constructor->ImageAdding(filtered_image , calibration.Tr_P2 );
            pcl::PointCloud<pcl::PointXYZ>::Ptr original_cloud
                    (new pcl::PointCloud<pcl::PointXYZ>());
            readLabel->ReadPCD(ls_filename , original_cloud , calibration.Tr_velo_cam);
            // Ground removal
            pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud
                    (new pcl::PointCloud<pcl::PointXYZ>());
            pcl::PointCloud<pcl::PointXYZ>::Ptr nonGround_cloud
                    (new pcl::PointCloud<pcl::PointXYZ>());
            pcl::ModelCoefficients::Ptr coef =
                    pcd_proc->Ground_Object_disscrimination
                    (original_cloud ,ground_cloud ,nonGround_cloud );
            // Clustering
            std::vector < pcl::PointCloud<pcl::PointXYZ>::Ptr > clusters =
                    pcd_proc->Clustering(nonGround_cloud);
            pcl::PointCloud<pcl::PointXYZ>::Ptr carPCD =
                    pcd_proc->CalcKeyPoints(clusters[1]); // 3 cyclist // 1 car
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr carPCD_color =
                    pcd_proc->ColorCloud(carPCD , filtered_image , calibration.Tr_P2);
            pcl::PointCloud<pcl::PointXYZ>::Ptr carPCD_MLS =
                    pcd_proc->MLSSurface(carPCD);
//            pcl::PointCloud<pcl::PointXYZRGB>::Ptr carPCD_MLS_color =
//                    pcd_proc->ColorCloud(carPCD_MLS , filtered_image , calibration.Tr_P2);
//            constructor->ReconstructMLS(carPCD_color,carPCD_MLS_color);

            constructor->Reconstruct(carPCD_color);

            std::string fName = "./Results/CAR.pcd";
            carPCD->width = 1;
            carPCD->height = carPCD->points.size();
            pcl::io::savePCDFileASCII(fName , *carPCD);

            std::string fNameMLS = "./Results/upsampled_CAR.pcd";
            carPCD_MLS->width = 1;
            carPCD_MLS->height = carPCD_MLS->points.size();
            pcl::io::savePCDFileASCII(fNameMLS , *carPCD_MLS);
            // STOP
            cv::imshow("data" , filtered_image);
            cv::waitKey(0);

        }
    }

    return a.exec();
}
