#include "Readlabels.h"

ReadingLabels::ReadingLabels()
{    
}

ReadingLabels::ReadingLabels(std::string TRN_VEL_DIR, std::string TRN_LCAM_DIR,std::string TRN_RCAM_DIR,std::string TRN_LBL_DIR,std::string TRN_CALIB_DIR)
{
    m_TRN_VEL_DIR = TRN_VEL_DIR;
    m_TRN_LCAM_DIR = TRN_LCAM_DIR;
    m_TRN_RCAM_DIR = TRN_RCAM_DIR;
    m_TRN_CALIB_DIR = TRN_CALIB_DIR;
    m_TRN_LBL_DIR = TRN_LBL_DIR;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr UpsampleRGB
(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud
            (new pcl::PointCloud<pcl::PointXYZRGB>());
    //std::cout << cloud->points.size() << std::endl;
    pcl::BilateralUpsampling<pcl::PointXYZRGB,pcl::PointXYZRGB> bu;
    //    std::cout << cloud->points[10] << std::endl;
    //    cloud->width = cloud->size();
    //    cloud->height = 1;
    //    std::cout << cloud->width << ":" << cloud->height << std::endl;
    //    pcl::PointCloud<pcl::PointXYZRGB> l;
    bu.setInputCloud(cloud);
    bu.setWindowSize(5);
    //bu.setSigmaColor();
    //bu.setSigmaDepth();
    bu.process(*out_cloud);
}
pcl::PointCloud<pcl::PointXYZ>::Ptr ReadingLabels::Upsample
(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud
            (new pcl::PointCloud<pcl::PointXYZ>());
    pcl::MovingLeastSquares<pcl::PointXYZ , pcl::PointXYZ> mls;
    mls.setInputCloud(cloud);
    mls.setSearchRadius(0.1);
    mls.setPolynomialFit(true);
    mls.setPolynomialOrder(2.0);
    mls.setUpsamplingMethod(
    pcl::MovingLeastSquares<pcl::PointXYZ,pcl::PointXYZ>::SAMPLE_LOCAL_PLANE);
    mls.setUpsamplingRadius(0.1);
    mls.setUpsamplingStepSize(0.08);
    mls.process(*out_cloud);
    return out_cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr ColorPointCloud
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
    //pcl::BilateralUpsampling<pcl::PointXYZ,pcl::PointXYZ> bu;

}

void ReadPCD(std::string filename , pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, bool filtered , cv::Mat Tr_velo_to_cam)
{
    int32_t num = 1000000;
    float* data = (float*) malloc (num * sizeof (float));
    //pointers
    float *px = data + 0;
    float *py = data + 1;
    float *pz = data + 2;
    float *pr = data + 3;
    // filename
    FILE* stream;
    stream = fopen(filename.c_str(),"rb");
    num = fread(data , sizeof(float) , num , stream )/ 4;

    for (int32_t i = 0; i < num ; i++)
    {
        pcl::PointXYZ p;
        p.x = *px; p.y = *py; p.z = *pz;
        cv::Mat p_vel(4,1,CV_64F,cv::Scalar(0));
        p_vel.at<double>(0,0) = p.x;
        p_vel.at<double>(1,0) = p.y;
        p_vel.at<double>(2,0) = p.z;
        p_vel.at<double>(3,0) = 1;
        cv::Mat p_cam(4,1,CV_64F,cv::Scalar(0));
        p_cam = Tr_velo_to_cam * p_vel;
        pcl::PointXYZ q;
        q.x = p_cam.at<double>(0,0) / p_cam.at<double>(3,0);
        q.y = p_cam.at<double>(1,0) / p_cam.at<double>(3,0);
        q.z = p_cam.at<double>(2,0) / p_cam.at<double>(3,0);
        // convert points from laser scannar coordinate system
        // into camera coordinate system
        if (filtered )
        {
            //filter angle
            float angle = atan2 (p.y,p.x) * 180 / 3.14159265359;
            if (std::abs(angle) < 45)
                // filter distance
                if ((p.x)*(p.x)+(p.y)*(p.y) < 100. * 100.)
                    //filter height
                    if (p.z > -2.0)
                    {
                        cloud->points.push_back(q);
                    }
        }
        else
        {
            cloud->points.push_back(q);
        }
        px+= 4 ; py += 4; pz +=4; pr+=4;
    }
    fclose(stream);
    free(data);
    return;
}

void ReadingLabels::ReadPCD(std::string filename , pcl::PointCloud<pcl::PointXYZ>::Ptr cloud , cv::Mat Tr_velo_to_cam)
{
    bool filtered = true;
    int32_t num = 1000000;
    float* data = (float*) malloc (num * sizeof (float));
    //pointers
    float *px = data + 0;
    float *py = data + 1;
    float *pz = data + 2;
    float *pr = data + 3;
    // filename
    FILE* stream;
    stream = fopen(filename.c_str(),"rb");
    num = fread(data , sizeof(float) , num , stream )/ 4;

    for (int32_t i = 0; i < num ; i++)
    {
        pcl::PointXYZ p;
        p.x = *px; p.y = *py; p.z = *pz;
        cv::Mat p_vel(4,1,CV_64F,cv::Scalar(0));
        p_vel.at<double>(0,0) = p.x;
        p_vel.at<double>(1,0) = p.y;
        p_vel.at<double>(2,0) = p.z;
        p_vel.at<double>(3,0) = 1;
        cv::Mat p_cam(4,1,CV_64F,cv::Scalar(0));
        p_cam = Tr_velo_to_cam * p_vel;
        pcl::PointXYZ q;
        q.x = p_cam.at<double>(0,0) / p_cam.at<double>(3,0);
        q.y = p_cam.at<double>(1,0) / p_cam.at<double>(3,0);
        q.z = p_cam.at<double>(2,0) / p_cam.at<double>(3,0);
        // convert points from laser scannar coordinate system
        // into camera coordinate system
        if (filtered )
        {
            //filter angle
            float angle = atan2 (p.y,p.x) * 180 / 3.14159265359;
            if (std::abs(angle) < 45)
                // filter distance
                if ((p.x)*(p.x)+(p.y)*(p.y) < 100. * 100.)
                    //filter height
                    if (p.z > -2.0)
                    {
                        cloud->points.push_back(q);
                    }
        }
        else
        {
            cloud->points.push_back(q);
        }
        px+= 4 ; py += 4; pz +=4; pr+=4;
    }
    fclose(stream);
    free(data);
    return;
}

void Object_Segmentation (std::string ls_filename , std::string img_filename , std::vector <Label> lbls , Calib calibration)
{
    cv::Mat lImage = cv::imread(img_filename ,cv::IMREAD_COLOR);
    pcl::PointCloud<pcl::PointXYZ>::Ptr original_cloud
            (new pcl::PointCloud<pcl::PointXYZ>());
    ReadPCD(ls_filename , original_cloud , true , calibration.Tr_velo_cam);
    //pcl::PointCloud<pcl::PointXYZ>::Ptr highRes_cloud = Upsample(original_cloud);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = ColorPointCloud
            (original_cloud, lImage, calibration.Tr_P2);
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =UpsampleRGB(cloud1);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("test"));
    viewer->setBackgroundColor(1.0,1.0,1.0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud,rgb,"test");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3,"test");
    for (int i = 0; i < lbls.size() ; i++)
    {
        Label obj_label = lbls[i];
        if (obj_label.object_type == "DontCare")
            continue;
        //image
        cv::Rect bb2D;
        bb2D.x = obj_label.bbox(0);
        bb2D.y = obj_label.bbox(1);
        bb2D.width = obj_label.bbox(2)-obj_label.bbox(0);
        bb2D.height = obj_label.bbox(3)-obj_label.bbox(1);
        cv::rectangle(lImage , bb2D,cv::Scalar(0,0,0));
        //LS
        //std::cout << "dimension: " << obj_label.dimensions << std::endl;
        double h = obj_label.dimensions(0);
        double w = obj_label.dimensions(1);
        double l = obj_label.dimensions(2);
        double ry = obj_label.rotation_y;
        //
        cv::Mat R = (cv::Mat_<double>(3,3) << cos(ry) , 0 , sin(ry),
                     0    , 1 ,  0,
                     -sin(ry), 0 , cos(ry));
        cv::Mat x_corner = (cv::Mat_<double>(1,8) <<
                            l/2 , l/2 ,-l/2 ,-l/2 , l/2 , l/2 ,-l/2 ,-l/2);
        cv::Mat y_corner = (cv::Mat_<double>(1,8) <<
                            0 , 0 ,0 ,0 , -h , -h ,-h ,-h);
        cv::Mat z_corner = (cv::Mat_<double>(1,8) <<
                            w/2 , -w/2 ,-w/2 ,w/2 , w/2 , -w/2 ,-w/2 ,w/2);
        cv::Mat corners;
        cv::vconcat(x_corner,y_corner, corners);
        cv::vconcat(corners,z_corner, corners);
        //clipper
        pcl::CropBox<pcl::PointXYZ> clipper;
        //std::cout << " type = " << obj_label.object_type << std::endl;
        //std::cout << " location = " << obj_label.location << std::endl;
        //Eigen::Vector3f t; t(0) = obj_label.location(2); t(1) = -obj_label.location(1); t(2) = -obj_label.location(0);
        clipper.setTranslation (obj_label.location);
        //                        q.x = -p.y; q.y = -p.z; q.z = p.x;
        Eigen::Vector3f r;
        r(0) = 0;
        r(1) = ry;
        r(2) = 0;
        //        clipper.setRotation(r);
        //        clipper.setMin(-Eigen::Vector4f(l/2, w/2, 0, 0));
        //        clipper.setMax(Eigen::Vector4f(l/2, w/2, h, 0));
        //        pcl::PointCloud<pcl::PointXYZ>::Ptr outcloud
        //                (new pcl::PointCloud<pcl::PointXYZ>());
        //        clipper.setInputCloud(cloud);
        //        clipper.filter(*outcloud);
        Eigen::Quaternionf q;q.x()=0;q.y()=0;q.z()=0;q.w()=1;
        //q(0) = 0; //q(1,1) = 0; q(2,2) = 0; q(3,3) = 1.0;
        std::string cubeID = "cube" + boost::lexical_cast<std::string> (i);
        viewer->addCube(obj_label.location,q,w,h,l,cubeID);
        viewer->setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR , 1.0,0.0,0.0,cubeID);
        //
        pcl::PointXYZ p;
        p.x = obj_label.location(0);
        p.y = obj_label.location(1);
        p.z = obj_label.location(2);
        std::string sphereID = "sphere" + boost::lexical_cast<std::string> (i);
        //viewer->addSphere(p,1.0,sphereID);
        //std::cout << "outcloud size = " << cloud->size() << std::endl;
        std::string fName = "cloud.pcd" ;
        cloud->width = 1;
        cloud->height = cloud->points.size();
        pcl::io::savePCDFileASCII(fName , *cloud);
        cloud->width = 1;
        cloud->height = cloud->points.size();

    }
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
    cv::namedWindow("Test");
    cv::imshow ("Test" , lImage);
    cv::waitKey(0);
}

std::string ReadingLabels::FindDatasetFileName(int datasetNo ,
                                               std::string  TRN_DIR )
{
    // iterator on dataset folder name in training folder
    for (boost::filesystem::directory_iterator itr(TRN_DIR);
         itr!=boost::filesystem::directory_iterator(); ++itr)
    {
        std::string TRN_LBL = boost::lexical_cast<std::string>
                (itr->path().c_str()) ;
        std::string TRN_LBL_File = TRN_LBL.substr(TRN_LBL.length()-4,TRN_LBL.length());
        // comparing dataset numebr and dataset folder name
        if (atoi(TRN_LBL_File.c_str()) == datasetNo)
            return TRN_LBL_File ;
    }
    return "";
}

std::string ReadingLabels::FindEpochFileName(int epochNo , std::string  TRN_DIR )
{
    // iterator on datasetf file name in training folder
    for (boost::filesystem::directory_iterator itr(TRN_DIR);
         itr!=boost::filesystem::directory_iterator(); ++itr)
    {
        std::string FNAME = boost::lexical_cast<std::string>
                (itr->path().filename().c_str()) ;
        if ((FNAME=="." )||(FNAME==".."))
            continue;
        // comparing epoch numebr and dataset file name
        int file_number = atoi(FNAME.c_str());
        if ((file_number==epochNo) && (FNAME.substr(0,2)!="._"  ))
            return boost::lexical_cast<std::string> (itr->path().c_str()) ;
    }
    return "";
}


std::vector<Label> ReadingLabels::ReadLabels(std::string filename , int epoch)
{
    std::vector<Label> labels;
    std::string line;
    std::ifstream file;
    file.open(filename.c_str() );
    if (file.is_open())
    {
        while(getline(file, line))
        {
            std::vector<std::string> strs;
            boost::split(strs, line, boost::is_any_of(" "));
            // construct label
            if (atoi(strs[0].c_str()) != epoch)
                continue;
            Label lbl;
            lbl.frame = atoi(strs[0].c_str());
            lbl.track_id = atoi(strs[1].c_str());
            lbl.object_type = strs[2].c_str();
            lbl.truncated = atoi(strs[3].c_str());
            lbl.occluded = atoi(strs[4].c_str());
            lbl.alpha = atof(strs[5].c_str());
            lbl.bbox(0) = atof(strs[6].c_str());
            lbl.bbox(1) = atof(strs[7].c_str());
            lbl.bbox(2) = atof(strs[8].c_str());
            lbl.bbox(3) = atof(strs[9].c_str());
            lbl.dimensions(0) = atof(strs[10].c_str());
            lbl.dimensions(1) = atof(strs[11].c_str());
            lbl.dimensions(2) = atof(strs[12].c_str());
            lbl.location(0) = atof(strs[13].c_str());
            lbl.location(1) = atof(strs[14].c_str());
            lbl.location(2) = atof(strs[15].c_str());
            lbl.rotation_y = atof(strs[16].c_str());
            //lbl.score = atof(strs[17].c_str());
            labels.push_back(lbl);
        }
        file.close();
    }
    return labels;
}

std::vector < std::vector<Label> > ReadingLabels::ReadLabelMap(std::string filename,int numOfEpochs)
{
    std::vector< std::vector<Label> >labels_map(numOfEpochs);
    std::vector<Label> labels;
    std::string line;
    std::ifstream file;
    file.open(filename.c_str() );
    if (file.is_open())
    {
        while(getline(file, line))
        {
            std::vector<std::string> strs;
            boost::split(strs, line, boost::is_any_of(" "));
            // construct label
            Label lbl;
            lbl.frame = atoi(strs[0].c_str());
            lbl.track_id = atoi(strs[1].c_str());
            lbl.object_type = strs[2].c_str();
            lbl.truncated = atoi(strs[3].c_str());
            lbl.occluded = atoi(strs[4].c_str());
            lbl.alpha = atof(strs[5].c_str());
            lbl.bbox(0) = atof(strs[6].c_str());
            lbl.bbox(1) = atof(strs[7].c_str());
            lbl.bbox(2) = atof(strs[8].c_str());
            lbl.bbox(3) = atof(strs[9].c_str());
            lbl.dimensions(0) = atof(strs[10].c_str());
            lbl.dimensions(1) = atof(strs[11].c_str());
            lbl.dimensions(2) = atof(strs[12].c_str());
            lbl.location(0) = atof(strs[13].c_str());
            lbl.location(1) = atof(strs[14].c_str());
            lbl.location(2) = atof(strs[15].c_str());
            lbl.rotation_y = atof(strs[16].c_str());
            //lbl.score = atof(strs[17].c_str());
            //std::cout << lbl.frame << std::endl;
            labels_map[lbl.frame].push_back(lbl);
            labels.push_back(lbl);
        }
        file.close();
    }
    return labels_map;
}

Calib ReadingLabels::ReadCalibs(std::string calib_filename)
{
    Calib calibration;
    std::string line;
    std::ifstream file;
    file.open(calib_filename.c_str() );
    if (file.is_open())
    {
        //std::cout << calib_filename << std::endl;
        while(getline(file, line))
        {
            //std::cout << line << std::endl;
            std::vector<std::string> strs;
            boost::split(strs, line, boost::is_any_of(" "));
            // construct label
            if ((strs[0] == "R_rect") )
            {
                cv::Mat mat44(4,4 , CV_64F,cv::Scalar(0));
                mat44.at<double>(0,0) = atof(strs[1].c_str());
                mat44.at<double>(0,1) = atof(strs[2].c_str());
                mat44.at<double>(0,2) = atof(strs[3].c_str());
                mat44.at<double>(1,0) = atof(strs[4].c_str());
                mat44.at<double>(1,1) = atof(strs[5].c_str());
                mat44.at<double>(1,2) = atof(strs[6].c_str());
                mat44.at<double>(2,0) = atof(strs[7].c_str());
                mat44.at<double>(2,1) = atof(strs[8].c_str());
                mat44.at<double>(2,2) = atof(strs[9].c_str());
                mat44.at<double>(3,3) = 1.0;
                calibration.R_rect = mat44;
            }
            else if((strs[0] == "Tr_velo_cam") ||(strs[0] == "Tr_imu_velo"))
            {
                cv::Mat mat44(4,4 , CV_64F,cv::Scalar(0));
                mat44.at<double>(0,0) = atof(strs[1].c_str());
                mat44.at<double>(0,1) = atof(strs[2].c_str());
                mat44.at<double>(0,2) = atof(strs[3].c_str());
                mat44.at<double>(0,3) = atof(strs[4].c_str());
                mat44.at<double>(1,0) = atof(strs[5].c_str());
                mat44.at<double>(1,1) = atof(strs[6].c_str());
                mat44.at<double>(1,2) = atof(strs[7].c_str());
                mat44.at<double>(1,3) = atof(strs[8].c_str());
                mat44.at<double>(2,0) = atof(strs[9].c_str());
                mat44.at<double>(2,1) = atof(strs[10].c_str());
                mat44.at<double>(2,2) = atof(strs[11].c_str());
                mat44.at<double>(2,3) = atof(strs[12].c_str());
                mat44.at<double>(3,3) = 1.0;
                if (strs[0] == "Tr_velo_cam")
                    calibration.Tr_velo_cam = mat44;
                if (strs[0] == "Tr_imu_velo")
                    calibration.Tr_imu_velo = mat44;
            }
            else
            {
                cv::Mat mat34(3,4 , CV_64F,cv::Scalar(0));
                mat34.at<double>(0,0) = atof(strs[1].c_str());
                mat34.at<double>(0,1) = atof(strs[2].c_str());
                mat34.at<double>(0,2) = atof(strs[3].c_str());
                mat34.at<double>(0,3) = atof(strs[4].c_str());
                mat34.at<double>(1,0) = atof(strs[5].c_str());
                mat34.at<double>(1,1) = atof(strs[6].c_str());
                mat34.at<double>(1,2) = atof(strs[7].c_str());
                mat34.at<double>(1,3) = atof(strs[8].c_str());
                mat34.at<double>(2,0) = atof(strs[9].c_str());
                mat34.at<double>(2,1) = atof(strs[10].c_str());
                mat34.at<double>(2,2) = atof(strs[11].c_str());
                mat34.at<double>(2,3) = atof(strs[12].c_str());
                if (strs[0] == "P0:")
                    calibration.P0 = mat34;
                else if (strs[0] == "P1:")
                    calibration.P1 = mat34;
                else if (strs[0] == "P2:")
                    calibration.P2 = mat34;
                else if (strs[0] == "P3:")
                    calibration.P3 = mat34;
                else
                    std::cerr << strs[0] << " Something is wrong " << std::endl;
            }
        }
        file.close();
    }
    calibration.Tr_P2 = calibration.P2 * calibration.R_rect ;
    calibration.Tr_P3 = calibration.P3 * calibration.R_rect ;
    return calibration;
}

int ReadingLabels::findEpochs(std::string TRN_VEL_DIR ,int datasetNo)
{
    int i=0;
    std::string dataset = FindEpochFileName(datasetNo , TRN_VEL_DIR);
    for (boost::filesystem::directory_iterator itr(dataset);
         itr!=boost::filesystem::directory_iterator(); ++itr)
    {
        std::string filename_with_ext = itr->path().filename().c_str();
        if ((filename_with_ext=="." )||(filename_with_ext==".."))
            continue;
        else
            i++;
    }
    return i;
}

std::vector<std::string> ReadingLabels::Read_Lbl_Calib(int datasetNo)
{
    std::vector < std::string> files;
    //std::cout << "datasetNo " << datasetNo << std::endl;
    std::string lbl_dataset = FindEpochFileName(datasetNo , m_TRN_LBL_DIR);
    //std::cout << "lbl_dataset " << lbl_dataset << std::endl;
    std::string calib_dataset = FindEpochFileName(datasetNo , m_TRN_CALIB_DIR);
    //std::cout << "calib_dataset " << calib_dataset << std::endl;
    files.push_back(lbl_dataset);
    files.push_back(calib_dataset);
    return files;
}


std::vector<std::string> ReadingLabels::Read_LS_Img(int datasetNo, int EpochNo)
{
    std::vector < std::string> files;
    std::string ls_dataset = FindEpochFileName(datasetNo , m_TRN_VEL_DIR);
    std::string ls_file = FindEpochFileName(EpochNo , ls_dataset);
    //std::cout << "ls_dataset " << ls_file << std::endl;
    std::string imgL_dataset = FindEpochFileName(datasetNo , m_TRN_LCAM_DIR);
    std::string imgR_dataset = FindEpochFileName(datasetNo , m_TRN_RCAM_DIR);
    std::string imgL_file = FindEpochFileName(EpochNo , imgL_dataset);
    std::string imgR_file = FindEpochFileName(EpochNo , imgR_dataset);
    //std::cout << "img_dataset " << img_file << std::endl;
    files.push_back(ls_file);
    files.push_back(imgL_file);
    files.push_back(imgR_file);
    return files;
}



std::vector<object_List> ReadingLabels::ReadData
(std::string lbl_dir, std::string ls_dir , std::string img_dir , std::string calib_dir)
{
    std::vector<object_List> allObjects;
    for (boost::filesystem::directory_iterator itr_lbl(lbl_dir);
         itr_lbl!=boost::filesystem::directory_iterator(); ++itr_lbl)
    {
        std::string lbl_filename_with_ext = itr_lbl->path().filename().c_str();
        if ((lbl_filename_with_ext=="." )||(lbl_filename_with_ext==".."))
            continue;
        std::string lbl_filename_NO_ext = lbl_filename_with_ext.substr(0,lbl_filename_with_ext.size()-4);
        int datasetNo = atoi(lbl_filename_NO_ext.c_str());
        if (datasetNo != 19)
            continue;
        std::string calib_dataset = FindEpochFileName(datasetNo , calib_dir);
        Calib calibration = ReadCalibs(calib_dataset );
        std::string ls_dataset = FindDatasetFileName(datasetNo , ls_dir);
        std::string ls_dataset_full = ls_dir + "/" + ls_dataset;
        std::string img_dataset = FindDatasetFileName(datasetNo , img_dir);
        std::string img_dataset_full = img_dir + "/" + img_dataset;
        std::string lbl_dataset_full = lbl_dir + "/" + lbl_filename_with_ext;
        for (boost::filesystem::directory_iterator itr_ls(ls_dataset_full);
             itr_ls!=boost::filesystem::directory_iterator(); ++itr_ls)
        {
            std::string ls_filename_with_ext = itr_ls->path().filename().c_str();
            if ((ls_filename_with_ext=="." )||(ls_filename_with_ext==".."))
                continue;
            std::string ls_filename_NO_ext = ls_filename_with_ext.substr(0,ls_filename_with_ext.size()-4);
            int epochNo = atoi(ls_filename_NO_ext.c_str());
            //            if (datasetNo != 1)
            //                continue;
            //std::cout << epochNo << ":" << img_dir << std::endl;
            std::string img_filename = FindEpochFileName(epochNo , img_dataset_full);
            std::vector<Label> labels = ReadLabels( lbl_dataset_full , epochNo);
            std::string ls_filename = ls_dataset_full + "/" + ls_filename_with_ext;
            std::cout << "img_filename: " << img_filename << std::endl;
            std::cout << "ls_filename: " << ls_filename << std::endl;
            std::cout << "lbl_dataset_full: " << lbl_dataset_full << std::endl;
            std::cout << "calib_dataset: " << calib_dataset << std::endl;
            Object_Segmentation ( ls_filename , img_filename , labels , calibration);
            object_List objectsInEpoch;
            objectsInEpoch.labels = labels;
            objectsInEpoch.img_filename = img_filename;
            objectsInEpoch.ls_filename = ls_filename;//ls_dataset_full + "/" + ls_filename_with_ext;
            objectsInEpoch.calibration = calibration;
            objectsInEpoch.lbl_filename = lbl_dataset_full;
            allObjects.push_back(objectsInEpoch);
            std::cout << "********" << std::endl;

        }
        std::cout << "___________________" << std::endl;

    }
    return allObjects;

}
