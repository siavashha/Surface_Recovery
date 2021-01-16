#include "proposed.h"

Proposed::Proposed()
{
}
//
void Proposed::ImageAdding(cv::Mat lImage , cv::Mat Tr)
{
    m_lImage = lImage;
    m_Tr = Tr;
}

// calculation tensor
cv::Mat Tensor2D(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloud)
{
    // Differentiation
    float dzx = 0 , dzy = 0;
    cv::Mat dX(cloud->size()-1 , 2 , CV_64F);

    for (int k = 1 ; k < cloud->size(); k++)
    {
        pcl::PointXYZRGB pnt = cloud->points[k];
        //std::cout  << "pnt:" << pnt << std::endl;
        if ((pnt.x == 0) || (pnt.y == 0))
            continue;
        dX.at<double>(k-1,0) = pnt.z / pnt.x;
        dX.at<double>(k-1,1) = pnt.z / pnt.y;
    }
    cv::Mat G(2 , 2 , CV_64F);
    //std::cout  << "cloud->size():" << (1. / cloud->size() ) << std::endl;

    G = (1. / cloud->size() ) * dX.t() * dX;
    //std::cout  << "G:" << G << std::endl;

    cv::GaussianBlur(G , G , cv::Size(5,5) , 0.5);
    G.convertTo(G , CV_64FC1);
    cv::Mat eigenValue(1,2,CV_64F);
    cv::Mat eigenVector(2,2,CV_64F);
    cv::eigen(G , eigenValue , eigenVector);
    cv::Mat eigenVector1(2,1,CV_64F);
    eigenVector1 = eigenVector.row(0).t();
    cv::Mat eigenVector2(2,1,CV_64F);
    eigenVector2 = eigenVector.row(1).t();
    double l1 = eigenValue.at<double>(0,0);
    double l2 = eigenValue.at<double>(0,1);
    double lambda1 = 1.0 / sqrt( l1 + l2 + 1 );
    double lambda2 = 1.0 / ( l1 + l2 + 1 );
    cv::Mat T = lambda1 * eigenVector2 * eigenVector2.t()
            + lambda2 * eigenVector1 * eigenVector1.t();
    return T;
}

// calculation tensor
cv::Mat Tensor3D(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloud)
{
    // Differentiation
    float dzx = 0 , dzy = 0;
    cv::Mat dX(cloud->size()-1 , 3 , CV_64F);

    for (int k = 1 ; k < cloud->size(); k++)
    {
        pcl::PointXYZRGB pnt = cloud->points[k];
        //std::cout  << "pnt:" << pnt << std::endl;
        if ((pnt.x == 0) || (pnt.y == 0))
            continue;
        dX.at<double>(k-1,0) = pnt.x;
        dX.at<double>(k-1,1) = pnt.y;
        dX.at<double>(k-1,2) = pnt.z;
    }
    cv::Mat G(3 , 3 , CV_64F);
    //std::cout  << "cloud->size():" << (1. / cloud->size() ) << std::endl;

    G = (1. / cloud->size() ) * dX.t() * dX;
    //std::cout  << "G:" << G << std::endl;

    cv::GaussianBlur(G , G , cv::Size(5,5) , 0.5);
    G.convertTo(G , CV_64FC1);
    cv::Mat eigenValue(1,3,CV_64F);
    cv::Mat eigenVector(3,3,CV_64F);
    cv::eigen(G , eigenValue , eigenVector);
    cv::Mat eigenVector1(3,1,CV_64F);
    eigenVector1 = eigenVector.row(0).t();
    cv::Mat eigenVector2(3,1,CV_64F);
    eigenVector2 = eigenVector.row(1).t();
    cv::Mat eigenVector3(3,1,CV_64F);
    eigenVector3 = eigenVector.row(2).t();
    double l1 = eigenValue.at<double>(0,0);
    double l2 = eigenValue.at<double>(0,1);
    double l3 = eigenValue.at<double>(0,2);
    double lambda1 = 1.0 / sqrt( l1 + l2 + l3 + 1 );
    double lambda2 = 1.0 / ( l1 + l2 + l3 + 1 );
    cv::Mat T = lambda1 * eigenVector2 * eigenVector2.t()
            + lambda2 * eigenVector1 * eigenVector1.t();
    return T;
}

cv::Mat Tensor3D(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloud , pcl::PointXYZRGB p)
{
    // Differentiation
    float dzx = 0 , dzy = 0;
    cv::Mat dX(cloud->size()-1 , 3 , CV_64F);

    for (int k = 0 ; k < cloud->size(); k++)
    {
        pcl::PointXYZRGB pnt = cloud->points[k];
        //std::cout  << "pnt:" << pnt << std::endl;
        if ((pnt.x == 0) || (pnt.y == 0))
            continue;
        dX.at<double>(k-1,0) = pnt.x - p.x;
        dX.at<double>(k-1,1) = pnt.y - p.y;
        dX.at<double>(k-1,2) = pnt.z - p.z;
    }
    cv::Mat G(3 , 3 , CV_64F);
    //std::cout  << "cloud->size():" << (1. / cloud->size() ) << std::endl;

    G = (1. / cloud->size() ) * dX.t() * dX;
    //std::cout  << "G:" << G << std::endl;

    cv::GaussianBlur(G , G , cv::Size(5,5) , 0.5);
    G.convertTo(G , CV_64FC1);
    cv::Mat eigenValue(1,3,CV_64F);
    cv::Mat eigenVector(3,3,CV_64F);
    cv::eigen(G , eigenValue , eigenVector);
    cv::Mat eigenVector1(3,1,CV_64F);
    eigenVector1 = eigenVector.row(0).t();
    cv::Mat eigenVector2(3,1,CV_64F);
    eigenVector2 = eigenVector.row(1).t();
    cv::Mat eigenVector3(3,1,CV_64F);
    eigenVector3 = eigenVector.row(2).t();
    double l1 = eigenValue.at<double>(0,0);
    double l2 = eigenValue.at<double>(0,1);
    double l3 = eigenValue.at<double>(0,2);
    double lambda1 = 1.0 / sqrt( l1 + l2 + l3 + 1 );
    double lambda2 = 1.0 / ( l1 + l2 + l3 + 1 );
    cv::Mat T = lambda1 * eigenVector2 * eigenVector2.t()
            + lambda2 * eigenVector1 * eigenVector1.t();
    return T;
}

// calculation normal
pcl::PointCloud<pcl::Normal>::Ptr CalcNormal(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloud)
{
    pcl::NormalEstimation<pcl::PointXYZRGB , pcl::Normal> ne;
    //ne.setKSearch(10);
    ne.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree
            ( new pcl::search::KdTree<pcl::PointXYZRGB> ());
    ne.setSearchMethod(tree);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals
            ( new pcl::PointCloud<pcl::Normal>);
    //    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_PointNormals
    //            ( new pcl::PointCloud<pcl::PointNormal>);
    ne.setRadiusSearch(0.4);
    ne.compute(*cloud_normals);
    std::vector<int> indices;
    //    pcl::removeNaNNormalsFromPointCloud(*cloud_normals, *cloud_normals , indices);
    //    pcl::concatenateFields(*cloud,*cloud_normals,*cloud_PointNormals);
    return cloud_normals;
}
// rotation matrix
//cv::Mat RotationMat(pcl::Normal l)
//{
//    cv::Mat rot1(3,3,CV_64F,cv::Scalar(0));
//    cv::Mat rot2(3,3,CV_64F,cv::Scalar(0));
//    cv::Mat rot(3,3,CV_64F,cv::Scalar(0));
//    double sinB = -l.normal_x;
//    double A = -atan2(l.normal_y , l.normal_z);
//    double cosB = std::sqrt(1 - l.normal_x * l.normal_x);
//    // rotation matrix 1
//    rot1.at<double>(0,0) = 1; rot1.at<double>(0,1) = 0;      rot1.at<double>(0,2) = 0;
//    rot1.at<double>(1,0) = 0; rot1.at<double>(1,1) = cos(A); rot1.at<double>(1,2) = sin(A);
//    rot1.at<double>(2,0) = 0; rot1.at<double>(2,1) = -sin(A); rot1.at<double>(2,2) = cos(A);
//    // rotation matrix 2
//    rot2.at<double>(0,0) = cosB; rot2.at<double>(0,1) = 0; rot2.at<double>(0,2) = -sinB;
//    rot2.at<double>(1,0) = 0;    rot2.at<double>(1,1) = 1; rot2.at<double>(1,2) = 0;
//    rot2.at<double>(2,0) = sinB; rot2.at<double>(2,1) = 0; rot2.at<double>(2,2) = cosB;
//    rot = rot1 * rot2;
//    return rot;
//}

// rotation matrix
cv::Mat RotationMat(pcl::Normal l)
{
    cv::Mat rot1(3,3,CV_64F,cv::Scalar(0));
    cv::Mat rot2(3,3,CV_64F,cv::Scalar(0));
    cv::Mat rot(3,3,CV_64F,cv::Scalar(0));
    double A = -atan2(l.normal_y , l.normal_z);
    double B = atan2(l.normal_x , std::sqrt( 1 - l.normal_x * l.normal_x));
    //ouble B = acos(std::sqrt(l.normal_y * l.normal_y + l.normal_z * l.normal_z) );
    //std::cout << " l = " << l << " ;A = " << A << " ;B = " << B << std::endl;
    double sinB = sin(B);//-l.normal_x;
    double cosB = cos(B);//std::sqrt(1 - l.normal_x * l.normal_x);
    // rotation matrix 1
    rot1.at<double>(0,0) = 1; rot1.at<double>(0,1) = 0;      rot1.at<double>(0,2) = 0;
    rot1.at<double>(1,0) = 0; rot1.at<double>(1,1) = cos(A); rot1.at<double>(1,2) = sin(A);
    rot1.at<double>(2,0) = 0; rot1.at<double>(2,1) = -sin(A); rot1.at<double>(2,2) = cos(A);
    // rotation matrix 2
    rot2.at<double>(0,0) = cosB; rot2.at<double>(0,1) = 0; rot2.at<double>(0,2) = -sinB;
    rot2.at<double>(1,0) = 0;    rot2.at<double>(1,1) = 1; rot2.at<double>(1,2) = 0;
    rot2.at<double>(2,0) = sinB; rot2.at<double>(2,1) = 0; rot2.at<double>(2,2) = cosB;
    rot = rot2 * rot1;

    cv::Mat x = (cv::Mat_<double>(3,1) << l.normal_x , l.normal_y , l.normal_z);

    //std::cout << " X = " << rot * x << std::endl;
    return rot.inv();
}

// rotation matrix
//cv::Mat RotationMat(pcl::Normal l)
//{
//    double c = (1 - l.normal_z) / (std::sqrt(l.normal_x * l.normal_x + l.normal_y * l.normal_y));
//    cv::Mat R(3,3,CV_64F,cv::Scalar(0));
//    R.at<double>(0,0)= 1 - (l.normal_x * l.normal_x) * c;
//    R.at<double>(0,1)= - (l.normal_x * l.normal_y) * c;
//    R.at<double>(0,2)= l.normal_x;
//    R.at<double>(1,0)= - (l.normal_x * l.normal_y) * c;
//    R.at<double>(1,1)= 1 - (l.normal_y * l.normal_y) * c;
//    R.at<double>(1,2)= l.normal_y;
//    R.at<double>(2,0)= - l.normal_x;
//    R.at<double>(2,1)= - l.normal_y;
//    R.at<double>(2,2)= 1 - (l.normal_x * l.normal_x + l.normal_y * l.normal_y) * c;

//    return R.inv();
//}

// Global sparse points
pcl::PointCloud<pcl::PointXYZRGB>::Ptr GlobalSparsePoints
(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud ,
 pcl::PointXYZRGB p , double& dist)
{
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(cloud);
    std::vector<int> knPntID;
    std::vector<float> knSqrDist;
    double radius = 0.3;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr neigh_cloud
            (new pcl::PointCloud<pcl::PointXYZRGB>());
    kdtree.radiusSearch(p , radius ,knPntID , knSqrDist );
    for (int k = 0 ; k < knPntID.size(); k++)
    {
        pcl::PointXYZRGB pnt = cloud->points[knPntID[k]];
        neigh_cloud->push_back(pnt);
    }
    if (knPntID.size() > 3)
        dist = 0.25 *
                (std::sqrt(knSqrDist[1]) + std::sqrt(knSqrDist[2]) +
                 std::sqrt(knSqrDist[3]) + std::sqrt(knSqrDist[4]));
    else if (knPntID.size() > 2)
        dist = 0.33 *
                (std::sqrt(knSqrDist[1]) + std::sqrt(knSqrDist[2]) +
                 std::sqrt(knSqrDist[3]));
    else if (knPntID.size() > 1)
        dist = 0.5 *
                (std::sqrt(knSqrDist[1]) + std::sqrt(knSqrDist[2]));
    else
        std::cout << knPntID.size() << " is not enough supporting points" << std::endl;
    if (std::isnan(dist))
        std::cout << " radius is nan" << std::endl;
    return neigh_cloud;
}
//// global to local
//pcl::PointCloud<pcl::PointXYZRGB>::Ptr Global2Local
//(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud ,
// pcl::PointXYZRGB origin, cv::Mat  rotation)
//{
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud
//            (new pcl::PointCloud<pcl::PointXYZRGB>());
//    for (int i = 0 ; i < cloud->size() ; i++)
//    {
//        pcl::PointXYZRGB p = cloud->points[i];
//        cv::Mat X = (cv::Mat_<double>(3,1) <<
//                     p.x - origin.x , p.y - origin.y , p.z - origin.z);
//        cv::Mat x = rotation * X;
//        p.x = x.at<double>(0,0);
//        p.y = x.at<double>(1,0);
//        p.z = x.at<double>(2,0);
//        out_cloud->push_back(p);
//    }
//    return out_cloud;
//}

// global to local
pcl::PointCloud<pcl::PointXYZRGB>::Ptr Global2Local
(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud ,
 pcl::PointXYZRGB origin, cv::Mat  rotation)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud
            (new pcl::PointCloud<pcl::PointXYZRGB>());
    for (int i = 0 ; i < cloud->size() ; i++)
    {
        pcl::PointXYZRGB p = cloud->points[i];
        cv::Mat X = (cv::Mat_<double>(3,1) <<
                     p.x - origin.x , p.y - origin.y , p.z - origin.z);
        cv::Mat x = rotation.inv() * X;
        p.x = x.at<double>(0,0);
        p.y = x.at<double>(1,0);
        p.z = x.at<double>(2,0);
        out_cloud->push_back(p);
    }
    return out_cloud;
}
//
// local dense grid points
//cv::Mat DepthMapGenerator
//(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud , double res ,
// cv::Mat& mask , pcl::PointCloud<pcl::PointXYZ>::Ptr& depth_cloud)
//{
//    pcl::PointXYZRGB p0 = cloud->points[0];
//    // Differentiation
//    cv::Mat A(cloud->size()-1 , 3 , CV_64F , cv::Scalar(0));
//    cv::Mat b(cloud->size()-1 , 1 , CV_64F , cv::Scalar(0));
//    cv::Mat S(cloud->size()-1 , cloud->size()-1 , CV_64F , cv::Scalar(0));
//    int x_dim = rint(dist / res) + 1;
//    int y_dim = rint(dist / res) + 1;
//    int radius = rint(rint(dist/res)/2);
//    for (int k = 1 ; k < cloud->size(); k++)
//    {
//        pcl::PointXYZRGB p = cloud->points[k];
//        double px = p.x - p0.x;
//        double py = p.y - p0.y;
//        double pz = p.z - p0.z;
//        A.at<double>(k-1,0) = px * px;
//        A.at<double>(k-1,1) = py * py;
//        A.at<double>(k-1,2) = px * py;
//        S.at<double>(k-1,k-1) = 1 / /*std::sqrt*/(px * px + py * py + pz * pz);
//        b.at<double>(k-1,0) = pz;
//    }
//    cv::Mat depthMap(x_dim , y_dim , CV_64F , cv::Scalar(0));
//    cv::Mat AAinv;
//    double condition_number = cv::invert(A.t()*S*A,AAinv,cv::DECOMP_SVD);
//    if (condition_number > 1e-1)
//    {
//        cv::Mat quad_coeff(3,1,CV_64F,cv::Scalar(0));
//        quad_coeff = AAinv * (A.t()*S*b);
//        for (int i = 0; i < x_dim ; i++)
//            for (int j = 0; j < y_dim ; j++)
//            {
//                double u = static_cast<double>(i - radius) * res;
//                double v = static_cast<double>(j - radius) * res;
//                double W = quad_coeff.at<double>(0,0) * u * u + quad_coeff.at<double>(1,0) * v * v +
//                        quad_coeff.at<double>(2,0) * u * v ;
//                depthMap.at<double>(i,j) = W;
//                pcl::PointXYZ p; p.x = u; p.y = v; p.z = W;
//                depth_cloud->points.push_back(p);
//            }
//    }
//    else
//        std::cout << "condition_number:" << condition_number << std::endl;
//    for (int k = 0 ; k < cloud->size() ; k++)
//    {
//        pcl::PointXYZRGB p = cloud->points[k];
//        int px = rint((p.x / res) + radius);
//        int py = rint((p.y / res) + radius);
//        if ((px > 0)&&(py>0)&&(px<depthMap.size().width)&&(py<depthMap.size().height))
//        {
//            depthMap.at<double>(px,py) = p.z;//p.z;
//            mask.at<double>(px,py) = 255;
//            //pcl::PointXYZ p; p.x = u; p.y = v; p.z = W;
//            depth_cloud->points.push_back(p);
//        }
//    }
//    return depthMap;
//}

//// estiamting second order
//cv::Mat DepthMap
//(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud , double dist , double res ,
// cv::Mat& mask , pcl::PointCloud<pcl::PointXYZ>::Ptr& depth_cloud)
//{
//    pcl::PointXYZRGB p0 = cloud->points[0];
//    // Differentiation
//    cv::Mat A(cloud->size()-1 , 5 , CV_64F , cv::Scalar(0));
//    cv::Mat C(cloud->size()-1 , 2 , CV_64F , cv::Scalar(0));
//    cv::Mat b(cloud->size()-1 , 1 , CV_64F , cv::Scalar(0));
//    cv::Mat S(cloud->size()-1 , cloud->size()-1 , CV_64F , cv::Scalar(0));
//    for (int k = 1 ; k < cloud->size(); k++)
//    {
//        pcl::PointXYZRGB p = cloud->points[k];
//        double px = p.x - p0.x;
//        double py = p.y - p0.y;
//        double pz = p.z - p0.z;
//        A.at<double>(k-1,0) = px * px;
//        A.at<double>(k-1,1) = py * py;
//        A.at<double>(k-1,2) = px * py;
//        A.at<double>(k-1,3) = px;
//        A.at<double>(k-1,4) = py;
//        C.at<double>(k-1,0) = px;
//        C.at<double>(k-1,1) = py;
//        S.at<double>(k-1,k-1) = 1 / std::sqrt(px * px + py * py + pz * pz);
//        b.at<double>(k-1,0) = pz;
//    }
//    int x_dim = rint(dist / res) + 1;
//    int y_dim = rint(dist / res) + 1;
//    int radius = rint(rint(dist/res)/2);
//    cv::Mat depthMap(x_dim , y_dim , CV_64F , cv::Scalar(0));
//    mask = cv::Mat::zeros(x_dim , y_dim , CV_64F);
//    cv::Mat AAinv;
//    double condition_number = cv::invert(A.t()*S*A,AAinv,cv::DECOMP_SVD);
//    cv::Mat CCinv;
//    double condition_number_linear = cv::invert(C.t()*S*C,CCinv,cv::DECOMP_SVD);
//    if (condition_number > 1e-3)
//    {
//        cv::Mat quad_coeff(5,1,CV_64F,cv::Scalar(0));
//        quad_coeff = AAinv * (A.t()*S*b);
//        for (int i = 0; i < x_dim ; i++)
//            for (int j = 0; j < y_dim ; j++)
//            {
//                double u = static_cast<double>(i - radius) * res;
//                double v = static_cast<double>(j - radius) * res;
//                double W = quad_coeff.at<double>(0,0) * u * u + quad_coeff.at<double>(1,0) * v * v +
//                        quad_coeff.at<double>(2,0) * u * v + quad_coeff.at<double>(3,0) * u +
//                        quad_coeff.at<double>(4,0) * v  ;
//                depthMap.at<double>(i,j) = W;
//                pcl::PointXYZ p; p.x = u; p.y = v; p.z = W;
//                depth_cloud->points.push_back(p);
//            }
//    }
//    else if ((condition_number_linear > 1e-3) || (condition_number < 1e-3))
//    {
//        std::cout << "condition_number_linear:" << condition_number_linear << std::endl;
//        cv::Mat liear_coeff(2,1,CV_64F,cv::Scalar(0));
//        liear_coeff = CCinv * (C.t() * S * b);
//        for (int i = 0; i < x_dim ; i++)
//            for (int j = 0; j < y_dim ; j++)
//            {
//                double u = static_cast<double>(i - radius) * res;
//                double v = static_cast<double>(j - radius) * res;
//                double W = liear_coeff.at<double>(0,0) * u +
//                        liear_coeff.at<double>(1,0) * v;
//                depthMap.at<double>(i,j) = W;
//                pcl::PointXYZ p; p.x = u; p.y = v; p.z = W;
//                depth_cloud->points.push_back(p);
//            }
//    }
//    else
//        std::cout << "condition_number:" << condition_number << std::endl;
//    for (int k = 0 ; k < cloud->size() ; k++)
//    {
//        pcl::PointXYZRGB p = cloud->points[k];
//        pcl::PointXYZ q; q.x = p.x; q.y = p.y; q.z = p.z;
//        int px = rint((p.x / res) + radius);
//        int py = rint((p.y / res) + radius);
//        if ((px > 0)&&(py>0)&&(px<depthMap.size().width)&&(py<depthMap.size().height))
//        {
//            depthMap.at<double>(px,py) = p.z;//p.z;
//            mask.at<double>(px,py) = 255;
//            depth_cloud->points.push_back(q);
//            //mask.at<uchar>(50,50) = 255;
//        }
//    }
//    return depthMap;
//}

// estiamting second order
cv::Mat DepthMap
(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud , double dist , double res ,
 cv::Mat& mask , pcl::PointCloud<pcl::PointXYZ>::Ptr& depth_cloud)
{
    int x_dim = rint(dist / res) + 1;
    int y_dim = rint(dist / res) + 1;
    int radius = rint(rint(dist/res)/2);
    cv::Mat depthMap(x_dim , y_dim , CV_64F , cv::Scalar(0));
    mask = cv::Mat::zeros(x_dim , y_dim , CV_64F);
    pcl::PointXYZRGB p0 = cloud->points[0];
    // Differentiation
    cv::Mat A(cloud->size()-1 , 3 , CV_64F , cv::Scalar(0));
    cv::Mat b(cloud->size()-1 , 1 , CV_64F , cv::Scalar(0));
    cv::Mat S(cloud->size()-1 , cloud->size()-1 , CV_64F , cv::Scalar(0));
    for (int k = 1 ; k < cloud->size(); k++)
    {
        pcl::PointXYZRGB p = cloud->points[k];
        double px = p.x - p0.x;
        double py = p.y - p0.y;
        double pz = p.z - p0.z;
        A.at<double>(k-1,0) = px * px;
        A.at<double>(k-1,1) = py * py;
        A.at<double>(k-1,2) = px * py;
        S.at<double>(k-1,k-1) = 1 / std::sqrt(px * px + py * py + pz * pz);
        b.at<double>(k-1,0) = pz;
    }

    cv::Mat AAinv;
    double condition_number = cv::invert(A.t()*S*A,AAinv,cv::DECOMP_SVD);
    if (condition_number > 1e-3)
    {
        cv::Mat quad_coeff(3,1,CV_64F,cv::Scalar(0));
        quad_coeff = AAinv * (A.t()*S*b);
        for (int i = 0; i < x_dim ; i++)
            for (int j = 0; j < y_dim ; j++)
            {
                double u = static_cast<double>(i - radius) * res;
                double v = static_cast<double>(j - radius) * res;
                double W = quad_coeff.at<double>(0,0) * u * u +
                        quad_coeff.at<double>(1,0) * v * v +
                        quad_coeff.at<double>(2,0) * u * v;
                depthMap.at<double>(i,j) = W;
                //                pcl::PointXYZ p; p.x = u; p.y = v; p.z = W;
                //                depth_cloud->points.push_back(p);
                pcl::PointXYZ q; q.x = u; q.y = v; q.z = W;
                depth_cloud->points.push_back(q);
            }
    }
    else
    {
        std::cout << "condition_number:" << condition_number << std::endl;
        for (int i = 0; i < x_dim ; i++)
            for (int j = 0; j < y_dim ; j++)
            {
                double u = static_cast<double>(i - radius) * res;
                double v = static_cast<double>(j - radius) * res;
                depthMap.at<double>(i,j) = 0;
                pcl::PointXYZ q; q.x = u; q.y = v; q.z = 0;
                depth_cloud->points.push_back(q);
            }
    }
    //    for (int k = 0 ; k < cloud->size() ; k++)
    //    {
    //        pcl::PointXYZRGB p = cloud->points[k];
    //        int px = rint((p.x / res) + radius);
    //        int py = rint((p.y / res) + radius);
    //        if ((px > 0)&&(py>0)&&(px<depthMap.size().width)&&(py<depthMap.size().height))
    //        {
    //            depthMap.at<double>(px,py) = p.z;//p.z;
    //            mask.at<double>(px,py) = 255;
    //            //mask.at<uchar>(50,50) = 255;
    //        }
    //    }
    return depthMap;
}
// local to global
pcl::PointCloud<pcl::PointXYZ>::Ptr Local2Global
(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud ,
 pcl::PointXYZRGB origin, cv::Mat rotation)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud
            (new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0 ; i < cloud->size() ; i++)
    {
        pcl::PointXYZ p = cloud->points[i];
        cv::Mat X = (cv::Mat_<double>(3,1) <<
                     p.x , p.y , p.z );
        cv::Mat x = rotation * X;
        p.x = x.at<double>(0,0) + origin.x;
        p.y = x.at<double>(1,0) + origin.y;
        p.z = x.at<double>(2,0) + origin.z;
        out_cloud->push_back(p);
    }
    return out_cloud;
}
// local to global
pcl::PointCloud<pcl::PointXYZRGB>::Ptr Local2Global
(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud ,
 pcl::PointXYZRGB origin, cv::Mat rotation)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud
            (new pcl::PointCloud<pcl::PointXYZRGB>);
    for (int i = 0 ; i < cloud->size() ; i++)
    {
        pcl::PointXYZRGB p = cloud->points[i];
        cv::Mat X = (cv::Mat_<double>(3,1) <<
                     p.x , p.y , p.z );
        cv::Mat x = rotation * X;
        p.x = x.at<double>(0,0) + origin.x;
        p.y = x.at<double>(1,0) + origin.y;
        p.z = x.at<double>(2,0) + origin.z;
        out_cloud->push_back(p);
    }
    return out_cloud;
}

// local dense grid points
cv::Mat EllipseMask
(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud , double dist , double res ,
 cv::Mat depth_Map , pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered_cloud)
{
    int x_dim = rint(dist / res) + 1;
    int y_dim = rint(dist / res) + 1;
    int radius = rint(rint(dist/res)/2);
    cv::Mat mask(x_dim,y_dim,CV_64F,cv::Scalar(0));
    pcl::PointXYZRGB p0 = cloud->points[0];
    // Differentiation
    cv::Mat dp(cloud->size()-1 , 2 , CV_64F , cv::Scalar(0));
    for (int k = 1 ; k < cloud->size(); k++)
    {
        pcl::PointXYZRGB p = cloud->points[k];
        double px = p.x - p0.x;
        double py = p.y - p0.y;
        dp.at<double>(k-1,0) = px;
        dp.at<double>(k-1,1) = py;
    }
    cv::Mat G = dp.t() * dp;
    // cv::GaussianBlur(G , G , cv::Size(5,5) , 0.5);
    G.convertTo(G , CV_64FC1);
    cv::Mat eigenValue(1,2,CV_64F);
    cv::Mat eigenVector(2,2,CV_64F);
    cv::eigen(G , eigenValue , eigenVector);
    cv::Mat eigenVector1(2,1,CV_64F);
    eigenVector1 = eigenVector.row(0).t();
    cv::Mat eigenVector2(2,1,CV_64F);
    eigenVector2 = eigenVector.row(1).t();
    double l1 = eigenValue.at<double>(0,0);
    double l2 = eigenValue.at<double>(0,1);
    // directions
    cv::Mat i_dir = (cv::Mat_<double>(2, 1) << 1 , 0 );
    cv::Mat j_dir = (cv::Mat_<double>(2, 1) << 0 , 1 );
    // dot products
    cv::Mat a11_mat = i_dir.t() * eigenVector1;
    double a11 = a11_mat.at<double>(0,0);
    cv::Mat a12_mat = i_dir.t() * eigenVector2;
    double a12 = a12_mat.at<double>(0,0);
    cv::Mat a21_mat = j_dir.t() * eigenVector1;
    double a21 = a21_mat.at<double>(0,0);
    cv::Mat a22_mat = j_dir.t() * eigenVector2;
    double a22 = a22_mat.at<double>(0,0);
    // coeffs
    double b_uu = (a11 / l1) * (a11 / l1) +
            (a12 / l2) * (a12 / l2) ;
    double b_vv = (a21 / l1) * (a21 / l1) +
            (a22 / l2) * (a22 / l2) ;
    double b_uv = 2*((a11 / l1) * (a21 / l1) +
                     (a12 / l2) * (a22 / l2));
    for (int i = 0; i < mask.size().height ; i ++)
        for (int j = 0; j < mask.size().width ; j ++)
        {
            double u = static_cast<double>(i - radius) * res;
            double v = static_cast<double>(j - radius) * res;
            // ellipse
            double ellipse = b_uu * u * u  + b_uv * u * v + b_vv * v * v;
            //std::cout << "coeff:" << ellipse << std::endl;
            if (ellipse < std::sqrt(2) * 1 * dist * dist)
            {
                mask.at<uchar>(i,j) = 255;
                pcl::PointXYZ q; q.x = u; q.y = v; q.z = depth_Map.at<double>(i,j);
                filtered_cloud->points.push_back(q);
            }
        }
    return mask;
}
// local cloud 2 local cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr
ColorCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud , cv::Mat image ,
           cv::Mat Tr , pcl::PointXYZRGB origin , cv::Mat rotation)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgbCloud
            (new pcl::PointCloud<pcl::PointXYZRGB>());
    for (int i = 0 ; i < cloud->size() ; i++)
    {
        pcl::PointXYZ p = cloud->points[i];
        cv::Mat Xl = (cv::Mat_<double>(3,1) <<
                      p.x , p.y , p.z );
        cv::Mat Xlr = rotation * Xl;
        cv::Mat Xg(4,1,CV_64F,cv::Scalar(0));
        Xg.at<double>(0,0) = Xlr.at<double>(0,0) + origin.x;
        Xg.at<double>(1,0) = Xlr.at<double>(1,0) + origin.y;
        Xg.at<double>(2,0) = Xlr.at<double>(2,0) + origin.z;
        Xg.at<double>(3,0) = 1;
        cv::Mat x(3,1,CV_64F,cv::Scalar(0));
        x = Tr * Xg;
        int qx = x.at<double>(0,0)/x.at<double>(2,0);
        int qy = x.at<double>(1,0)/x.at<double>(2,0);
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

pcl::PointCloud<pcl::PointXYZ>::Ptr geometry_regularizer
(       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud ,
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr lsc_cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud
            (new pcl::PointCloud<pcl::PointXYZ>());
    double max_dist = 0 , sigma = 1;
    pcl::PointXYZRGB pnt0 = lsc_cloud->points[0];
    for (int k = 1 ; k < lsc_cloud->size(); k++)
    {
        pcl::PointXYZRGB pnt = lsc_cloud->points[k];
        double distance = std::sqrt( (pnt.x - pnt0.x) * (pnt.x - pnt0.x) +
                                     (pnt.y - pnt0.y) * (pnt.y - pnt0.y) +
                                     (pnt.z - pnt0.z) * (pnt.z - pnt0.z));
        if (distance > max_dist)
            max_dist = distance;
    }
    double normalizer = max_dist / std::sqrt(3);
    for (int i = 1 ; i < cloud->size(); i++)
    {
        pcl::PointXYZRGB p = cloud->points[i];
        cv::Mat T = Tensor3D(lsc_cloud , p);
        cv::Mat Tinv;
        cv::invert(T,Tinv,cv::DECOMP_SVD);
        //std::cout << " Tinv = " << Tinv << std::endl;
        cv::Mat quadratic (1,1,CV_64F,cv::Scalar(0));
        double coef_sum = 0 , z_sum =0;
        for (int k = 0 ; k < lsc_cloud->size(); k++)
        {
            pcl::PointXYZRGB pnt = lsc_cloud->points[k];
            cv::Mat x = (cv::Mat_<double>(3,1) << pnt.x - p.x  , pnt.y - p.y ,
                         pnt.z - p.z ) / normalizer;
//            cv::Mat x = (cv::Mat_<double>(2,1) << pnt.x - p.x  , pnt.y - p.y) / normalizer;
            quadratic = -(x.t() * Tinv * x)/(2 * sigma );
            double quad = quadratic.at<double>(0,0);
            double coef = exp(quad);
            if (isnan(coef))
            {
                std::cout << "exponential is nan " << std::endl;
                continue;
            }
            //std::cout << coef << ":";
            z_sum = z_sum + pnt.z * coef;
            coef_sum = coef + coef_sum;
        }
        //std::cout << std::endl;
        double z = z_sum / coef_sum;
        pcl::PointXYZ p_out; p_out.x = p.x; p_out.y = p.y; p_out.z = p.z;
        out_cloud->points.push_back(p_out);
    }
    return out_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr brightness_regularizer(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud ,
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr lsc_cloud)
{
    //std::cout << " data " << std::endl;
    double K = 2;
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud
            (new pcl::PointCloud<pcl::PointXYZ>());
    for (int k = 0 ; k < cloud->size(); k++)
    {
        pcl::PointXYZRGB pnt = cloud->points[k];
        double br = 0.299 * pnt.r + 0.587 * pnt.g + 0.114 * pnt.b;
        double sum_coeff = 0;
        double pnt_z = 0;
        for (int i = 1 ; i < lsc_cloud->size(); i++)
        {
            pcl::PointXYZRGB p = lsc_cloud->points[i];
            double br0 = 0.299 * p.r + 0.587 * p.g + 0.114 * p.b;
            double d_br = ((std::abs(br - br0) + 1) / 256) * (M_PI / 2);
            //
            double dx = p.x - pnt.x ;
            double dy = p.y - pnt.y ;
            double dist = std::sqrt (dx * dx + dy * dy ) ;
            double dW_u = std::abs((p.z - pnt.z) /dx);
            double dW_v = std::abs((p.z - pnt.z) /dy);
            double dW = std::sqrt (dW_u * dW_u + dW_v * dW_v);
            if (isnan(dW))
                dW = std::abs(p.z - pnt.z) / dist;
            double coeff = exp(-std::pow(tan(d_br)- dW,2)/(K*K));
            if (isnan(coeff))
            {
                std::cout << "exponential is nan; " << std::pow(tan(d_br)- dW,2) <<
                             " tangent= " << tan(d_br) << " ;dW= " << dW << std::endl;
                continue;
            }

            //            std::cout << " tangent= " << tan(d_br) << " ;dW= " << dW <<
            //                         " ;coeff= " << coeff << std::endl;
            pnt_z = pnt_z + coeff * p.z;
            sum_coeff = sum_coeff + coeff;
            //D = (1 - tan(d_br) / dW) * exp( -std::pow(tan(d_br) - dW,2) / K * K);

            //(1 - d_br / dW)
        }
        double p_z = pnt_z / sum_coeff;
        pcl::PointXYZ q; q.x = pnt.x; q.y = pnt.y; q.z = p_z;
        out_cloud->points.push_back(q);
    }
    //std::cout << " out_cloud " << out_cloud->size() << std::endl;
    return out_cloud;
}
// outlier removal
pcl::PointCloud<pcl::PointXYZRGB>::Ptr OutlierRemoval(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered
            (new pcl::PointCloud<pcl::PointXYZRGB>());
    // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (5);
    sor.setStddevMulThresh (3.0);
    sor.filter (*cloud_filtered);
    return cloud_filtered;
}
// outlier removal
pcl::PointCloud<pcl::PointXYZRGB>::Ptr MedianFilter(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered
            (new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::MedianFilter<pcl::PointXYZRGB> mf;
    mf.setInputCloud (cloud);
    mf.setWindowSize(5);
    mf.setMaxAllowedMovement(0.000001);
    mf.applyFilter(*cloud_filtered);
    return cloud_filtered;
}
//Morphological
pcl::PointCloud<pcl::PointXYZRGB>::Ptr MorphologicalFilter(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud , double res)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered
            (new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::applyMorphologicalOperator<pcl::PointXYZRGB>(cloud,res,pcl::MORPH_CLOSE,*cloud_filtered);
    return cloud_filtered;
}

// Projection
pcl::PointCloud<pcl::PointXYZRGB>::Ptr Projection(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud , cv::Mat image, cv::Mat Tr)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud
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
        q.x = qx;
        q.y = qy;
        q.z = 0;
        uint32_t rgb = (static_cast<uint32_t>(red) << 16 |
                        static_cast<uint32_t>(green) << 8 | static_cast<uint32_t>(blue));
        q.rgb = *reinterpret_cast<float*>(&rgb);
        out_cloud->points.push_back(q);
    }
    return out_cloud;
}

// Reconstruction
pcl::PointCloud<pcl::PointXYZ>::Ptr Global_Regularizer(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud ,
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr lsc_poins , cv::Mat image, cv::Mat Tr)
{
    //std::cout << 0 << std::endl;
    cv::Mat mask1(image.size() , CV_64F, cv::Scalar(-10000));
    cv::Mat mask2(image.size() , CV_64F, cv::Scalar(-10000));
    cv::Mat mask3(image.size() , CV_64F, cv::Scalar(-10000));
    cv::Mat counter(image.size() , CV_8U, cv::Scalar(0));
    for (int32_t i = 0; i < cloud->size(); i++)
    {
        pcl::PointXYZRGB p;
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
        // color of p
        cv::Vec3b intensity = image.at< cv::Vec3b >(qy , qx);
        uchar blue = intensity.val[0];
        uchar green = intensity.val[1];
        uchar red = intensity.val[2];
        double br = 0.299 * red + 0.587 * green + 0.114 * blue;
        //std::cout << " 1 " << mask1.at<double>(qy,qx) << std::endl;
        if (mask1.at<double>(qy,qx)==-10000)
        {
            mask1.at<double>(qy,qx)=p.x;
            mask2.at<double>(qy,qx)=p.y;
            mask3.at<double>(qy,qx)=p.z;
            counter.at<char>(qy,qx) = 1;
        }
        else
        {
            //std::cout << 2 << std::endl;
            pcl::PointXYZ q;
            q.x = mask1.at<double>(qy,qx);
            q.y = mask2.at<double>(qy,qx);
            q.z = mask3.at<double>(qy,qx);
            pcl::PointXYZRGB q_rgb; q_rgb.x = q.x; q_rgb.y = q.y; q_rgb.z = q.z;
            q_rgb.r = 0; q_rgb.g = 0; q_rgb.b = 0;
            pcl::PointXYZRGB p_rgb; p_rgb.x = p.x; p_rgb.y = p.y; p_rgb.z = p.z;
            p_rgb.r = 0; p_rgb.g = 0; p_rgb.b = 0;
            // finding neighboring points of p and q
            double radius = 0.3;
            pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
            kdtree.setInputCloud(lsc_poins);
            std::vector<int> knPntID_p;
            std::vector<float> knSqrDist_p;
            std::vector<int> knPntID_q;
            std::vector<float> knSqrDist_q;
            kdtree.radiusSearch(p_rgb , radius ,knPntID_p , knSqrDist_p );
            kdtree.radiusSearch(q_rgb , radius ,knPntID_q , knSqrDist_q );
            // weight of p;
            double sum_coeff = 0 , pnt_x = 0 , pnt_y = 0 , pnt_z = 0, K =1;
            for (int k = 0 ; k < knPntID_p.size(); k++)
            {
                pcl::PointXYZRGB pnt_p = lsc_poins->points[knPntID_p[k]];
                double br0 = 0.299 * pnt_p.r + 0.587 * pnt_p.g + 0.114 * pnt_p.b;
                double d_br = ((std::abs(br - br0) + 1) / 256) * (M_PI / 2);
                //
                double dx = p.x - pnt_p.x ;
                double dy = p.y - pnt_p.y ;
                double dist = std::sqrt (dx * dx + dy * dy ) ;
                double dW_u = std::abs((p.z - pnt_p.z) /dx);
                double dW_v = std::abs((p.z - pnt_p.z) /dy);
                double dW = std::sqrt (dW_u * dW_u + dW_v * dW_v);
                if (isnan(dW))
                    dW = std::abs(p.z - pnt_p.z) / dist;
                double coeff = exp(-std::pow(tan(d_br)- dW,2)/(K*K));
                if (isnan(coeff))
                {
                    std::cout << "exponential is nan; " << std::pow(tan(d_br)- dW,2) <<
                                 " tangent= " << tan(d_br) << " ;dW= " << dW << std::endl;
                    continue;
                }
                //            std::cout << " tangent= " << tan(d_br) << " ;dW= " << dW <<
                //                         " ;coeff= " << coeff << std::endl;
                pnt_x = pnt_x + coeff * p.x;
                pnt_y = pnt_y + coeff * p.y;
                pnt_z = pnt_z + coeff * p.z;
                sum_coeff = sum_coeff + coeff;
            }
            //std::cout << 3 << std::endl;

            // weight of q;
            for (int k = 0 ; k < knPntID_q.size(); k++)
            {
                pcl::PointXYZRGB pnt_q = lsc_poins->points[knPntID_q[k]];
                double br0 = 0.299 * pnt_q.r + 0.587 * pnt_q.g + 0.114 * pnt_q.b;
                double d_br = ((std::abs(br - br0) + 1) / 256) * (M_PI / 2);
                //
                double dx = q.x - pnt_q.x ;
                double dy = q.y - pnt_q.y ;
                double dist = std::sqrt (dx * dx + dy * dy ) ;
                double dW_u = std::abs((q.z - pnt_q.z) /dx);
                double dW_v = std::abs((q.z - pnt_q.z) /dy);
                double dW = std::sqrt (dW_u * dW_u + dW_v * dW_v);
                if (isnan(dW))
                    dW = std::abs(q.z - pnt_q.z) / dist;
                double coeff = exp(-std::pow(tan(d_br)- dW,2)/(K*K));
                if (isnan(coeff))
                {
                    std::cout << "exponential is nan; " << std::pow(tan(d_br)- dW,2) <<
                                 " tangent= " << tan(d_br) << " ;dW= " << dW << std::endl;
                    continue;
                }
                //            std::cout << " tangent= " << tan(d_br) << " ;dW= " << dW <<
                //                         " ;coeff= " << coeff << std::endl;
                pnt_x = pnt_x + coeff * p.x;
                pnt_y = pnt_y + coeff * p.y;
                pnt_z = pnt_z + coeff * p.z;
                sum_coeff = sum_coeff + coeff;
                //std::cout << 4 << std::endl;

            }
            mask1.at<double>(qy,qx)=pnt_x / sum_coeff;
            mask2.at<double>(qy,qx)=pnt_y / sum_coeff;
            mask3.at<double>(qy,qx)=pnt_z / sum_coeff;
            counter.at<char>(qy,qx) = counter.at<char>(qy,qx) + 1;
        }
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud
            (new pcl::PointCloud<pcl::PointXYZ>());
    for (int i = 0 ; i < mask1.cols ; i++)
        for (int j = 0 ; j < mask1.rows ; j++)
            if(mask1.at<double>(j,i)!=-10000)
            {
                //std::cout << i << " / " <<  mask1.cols << std::endl;
                pcl::PointXYZ p;
                p.x = mask1.at<double>(j,i);
                p.y = mask2.at<double>(j,i);
                p.z = mask3.at<double>(j,i);
                out_cloud->points.push_back(p);
            }
    return out_cloud;
}

// Reconstruction
//pcl::PointCloud<pcl::PointXYZRGB>::Ptr Global_Regularizer2(
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud ,
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr lsc_poins , cv::Mat image, cv::Mat Tr , double min_res)
//{
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud
//            (new pcl::PointCloud<pcl::PointXYZRGB>());
//    pcl::PointCloud<pcl::Normal>::Ptr cloud_normal = CalcNormal(cloud);
//    for (int32_t i = 0; i < cloud->size(); i++)
//    {
//        std::cout << " p " << i << "/" << cloud->size() << std::endl;
//        pcl::PointXYZRGB p = cloud->points[i];
//        pcl::Normal l = cloud_normal->points[i];
//        cv::Mat n = (cv::Mat_<double>(3,1) << l.normal_x , l.normal_y , l.normal_z);
//        double radius = min_res * 10;
//        pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
//        kdtree.setInputCloud(cloud);
//        std::vector<int> knPntID;
//        std::vector<float> knSqrDist;
//        kdtree.radiusSearch(p, radius ,knPntID , knSqrDist);
//        double sum_xn = 0;
//        for (int j = 1; j < knPntID.size(); j++)
//        {
//            pcl::PointXYZRGB q = cloud->points[j];
//            cv::Mat x = (cv::Mat_<double>(3,1) << p.x - q.x , p.y - q.y, p.z - q.z);
//            cv::Mat xn(1,1,CV_64F,cv::Scalar(0));
//            xn = x.t() * n;
//            sum_xn = sum_xn + static_cast<double>(xn.at<double>(0,0));
//        }
//        double coeff = (1 / knPntID.size()) * sum_xn;
//        pcl::PointXYZRGB pnt; pnt.x = p.x + coeff * l.normal_x;
//        pnt.y = p.y + coeff * l.normal_y; pnt.z = p.z + coeff * l.normal_z; pnt.rgb = p.rgb;
//        out_cloud->points.push_back(pnt);
//    }
//    return out_cloud;
//}



// Reconstruction
pcl::PointCloud<pcl::PointXYZRGB>::Ptr Global_Regularizer1(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud ,
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr control_poins , cv::Mat image, cv::Mat Tr)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud
            (new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normal = CalcNormal(cloud);
    for (int32_t i = 0; i < cloud->size(); i++)
    {
        std::cout << " p " << i << "/" << cloud->size() << std::endl;
        pcl::PointXYZRGB p = cloud->points[i];
        pcl::Normal l = cloud_normal->points[i];
        cv::Mat n = (cv::Mat_<double>(3,1) << l.normal_x , l.normal_y , l.normal_z);
        double radius = 0.05;
        pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
        kdtree.setInputCloud(cloud);
        std::vector<int> knPntID;
        std::vector<float> knSqrDist;
        kdtree.radiusSearch(p, radius ,knPntID , knSqrDist);
        double sum_xn = 0 , sum_xx = 0;
        //std::cout << " knPntID_size() " << knPntID.size() << std::endl;

        for (int j = 0; j < knPntID.size(); j++)
        {
            int id = knPntID[j];
            pcl::PointXYZRGB q = cloud->points[id];
            cv::Mat x = (cv::Mat_<double>(3,1) << p.x - q.x , p.y - q.y, p.z - q.z);
            cv::Mat xn(1,1,CV_64F,cv::Scalar(0));
            cv::Mat xx(1,1,CV_64F,cv::Scalar(0));
            xn = x.t() * n;
            xx = x.t() * x;
            double xx_d = std::sqrt(xx.at<double>(0,0));
            double xn_d = xn.at<double>(0,0);
//            std::cout << " xn " << xn << " x = " << x << " xx = " << xx << std::endl;
//            std::cout << j << " xx_d " << xx_d << " knSqrDist = " << knSqrDist[j] << std::endl;
            sum_xn = sum_xn + xn_d;
        }
        //std::cout << knPntID.size() << " sum_xn " << sum_xn << std::endl;
        double coeff = -(1. / knPntID.size()) * sum_xn;
        //std::cout << " knPntID.size() " << knPntID.size() << " coeff " << coeff << std::endl;
        pcl::PointXYZRGB pnt; pnt.x = p.x + coeff * l.normal_x;
        pnt.y = p.y + coeff * l.normal_y; pnt.z = p.z + coeff * l.normal_z; pnt.rgb = p.rgb;
        out_cloud->points.push_back(pnt);
    }
    return out_cloud;
}

// Reconstruction
void Proposed::Reconstruct(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud )
{
    // normals
    pcl::PointCloud<pcl::Normal>::Ptr normals =
            CalcNormal(cloud);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_PointNormals
            ( new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_initial
            ( new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_brightness
            ( new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_brightness_2itr
            ( new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_geometry
            ( new pcl::PointCloud<pcl::PointXYZRGB>());
    //pcl::removeNaNNormalsFromPointCloud(*cloud_normals, *cloud_normals , indices);
    pcl::concatenateFields(*cloud,*normals,*cloud_PointNormals);
    std::string fName_noraml = "./Results/cloudRGBNormal.pcd";
    cloud_PointNormals->width = 1;
    cloud_PointNormals->height = cloud_PointNormals->points.size();
    pcl::io::savePCDFileASCII(fName_noraml , *cloud_PointNormals);

    for (int i  = 0 ; i < cloud->size() ; i++)
    {
        pcl::PointXYZRGB p = cloud->points[i];
        pcl::Normal l = normals->points[i];
        if (isnan(l.normal_x))
        {
            std::cout << " normal is nan " << std::endl;
            continue;
        }
        cv::Mat rotation = RotationMat(l);
        double dist = 0;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr gscPoints =
                GlobalSparsePoints(cloud , p , dist);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr lscPoints =
                Global2Local(gscPoints , p , rotation );
        if (std::isnan(dist))
        {
            std::cout << " dist is nan " << std::endl;
            continue;
        }
        dist = dist * 2;
        double res = 0.1 * dist;
        cv::Mat sampleMask;
        pcl::PointCloud<pcl::PointXYZ>::Ptr depth_cloud
                (new pcl::PointCloud<pcl::PointXYZ>());
        cv::Mat depthMap = DepthMap (lscPoints , dist , res , sampleMask , depth_cloud);
        //std::cout << "depth_cloud = " << depth_cloud->size() << std::endl;
        // Ellipse mask
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ell
                (new  pcl::PointCloud<pcl::PointXYZ>());
        cv::Mat ell_mask = EllipseMask(lscPoints , dist , res , depthMap , cloud_ell);

        //        std::cout << "cloud_ell = " << cloud_ell->size() << std::endl;

        //        // local to global
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud =
                ColorCloud( cloud_ell , m_lImage , m_Tr , p , rotation);
        std::cout << "colored_cloud = " << colored_cloud->size() << std::endl;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_patch_in = Local2Global
                (colored_cloud , p , rotation);

        for (int k_in = 0; k_in < global_patch_in->size() ; k_in++)
            cloud_initial->points.push_back(global_patch_in->points[k_in]);

        //        std::string fName = "./Results/Initial/patch" + boost::lexical_cast<std::string>(i)+ ".pcd";
        //        global_patch->width = 1;
        //        global_patch->height = global_patch->points.size();
        //        pcl::io::savePCDFileASCII(fName , *global_patch);

        pcl::PointCloud<pcl::PointXYZ>::Ptr br_regularizer_cloud =
                brightness_regularizer(colored_cloud , lscPoints);
        std::cout << "br_regularizer_cloud = " << br_regularizer_cloud->size() << std::endl;
        //
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr br_regularizer_cloud_g =
//                Global_Regularizer(br_regularizer_cloud , lscPoints , m_lImage , m_Tr);


        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud_br =
                ColorCloud( br_regularizer_cloud , m_lImage , m_Tr , p , rotation);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_patch_br = Local2Global
                (colored_cloud_br , p , rotation);
        std::cout << "global_patch_br = " << global_patch_br->size() << std::endl;

        for (int k_br1 = 0; k_br1 < global_patch_br->size() ; k_br1++)
            cloud_brightness->points.push_back(global_patch_br->points[k_br1]);

        std::cout << "cloud  " << cloud_brightness->size() << std::endl;

        // second iteration
        pcl::PointCloud<pcl::PointXYZ>::Ptr br_regularizer_cloud_2iter =
                brightness_regularizer(colored_cloud_br , lscPoints);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud_br_2itr =
                ColorCloud( br_regularizer_cloud_2iter , m_lImage , m_Tr , p , rotation);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_patch_br_2itr = Local2Global
                (colored_cloud_br_2itr , p , rotation);

        for (int k_br1 = 0; k_br1 < global_patch_br_2itr->size() ; k_br1++)
            cloud_brightness_2itr->points.push_back(global_patch_br_2itr->points[k_br1]);


        ///////////////////////////////////////
        ///////////////////////////////////////

        pcl::PointCloud<pcl::PointXYZ>::Ptr regularizer_cloud_geo =
                geometry_regularizer(colored_cloud , lscPoints);
        std::cout << "geo_regularizer_cloud = " << regularizer_cloud_geo->size() << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud_geo =
                ColorCloud( regularizer_cloud_geo , m_lImage , m_Tr , p , rotation);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_patch_geo = Local2Global
                (colored_cloud_geo , p , rotation);
        std::cout << "global_patch_geo = " << global_patch_geo->size() << std::endl;

        for (int k_br2 = 0; k_br2 < global_patch_geo->size() ; k_br2++)
            cloud_geometry->points.push_back(global_patch_geo->points[k_br2]);

        std::cout << "cloud_geometry  " << cloud_geometry->size() << std::endl;

        //        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud_br =
        //        ColorCloud( br_regularizer_cloud , m_lImage , m_Tr , p , rotation);
        ////        std::cout << "colored_cloud = " << colored_cloud->size() << std::endl;

        //        pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_patch_br = Local2Global
        //                (colored_cloud_br , p , rotation);

        //        std::string fName1 = "./Results/Brightness/patch" + boost::lexical_cast<std::string>(i)+ ".pcd";
        //        global_patch_br->width = 1;
        //        global_patch_br->height = global_patch_br->points.size();
        //        pcl::io::savePCDFileASCII(fName1 , *global_patch_br);




        //std::cout << "br_regularizer_cloud = " << br_regularizer_cloud->size() << std::endl;

        //        pcl::PointCloud<pcl::PointXYZ>::Ptr global_patch = Local2Global
        //                (cloud_ell , p , rotation);
        //        std::string fName = "./Results/patch" + boost::lexical_cast<std::string>(i)+ ".pcd";
        //        global_patch->width = 1;
        //        global_patch->height = global_patch->points.size();
        //        pcl::io::savePCDFileASCII(fName , *global_patch);

        //        pcl::PointCloud<pcl::PointXYZ>::Ptr global_patch_br = Local2Global
        //                (br_regularizer_cloud , p , rotation);
        //        std::string fName = "./Results/patch_br" + boost::lexical_cast<std::string>(i)+ ".pcd";
        //        global_patch_br->width = 1;
        //        global_patch_br->height = global_patch_br->points.size();
        //        pcl::io::savePCDFileASCII(fName , *global_patch_br);

        //
        std::cout << " point: " << i << "/" << cloud->size() << ",Radius=" << dist << std::endl;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr br_regularizer_cloud_g =
            Global_Regularizer(cloud_brightness , cloud , m_lImage , m_Tr);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr br_regularizer_cloud_g1 =
            Global_Regularizer1(cloud_brightness , cloud , m_lImage , m_Tr);


    // Initialization
    std::string fName_in = "./Results/initial.pcd";
    cloud_initial->width = 1;
    cloud_initial->height = cloud_initial->points.size();
    pcl::io::savePCDFileASCII(fName_in , *cloud_initial);

    // Brightness
    std::string fName_br = "./Results/brightness.pcd";
    cloud_brightness->width = 1;
    cloud_brightness->height = cloud_brightness->points.size();
    pcl::io::savePCDFileASCII(fName_br , *cloud_brightness);

    // Brightness 2 iteration
    std::string fName_br_2itr = "./Results/brightness2.pcd";
    cloud_brightness_2itr->width = 1;
    cloud_brightness_2itr->height = cloud_brightness_2itr->points.size();
    pcl::io::savePCDFileASCII(fName_br_2itr , *cloud_brightness_2itr);

//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud = MorphologicalFilter(cloud_brightness , 0.1 * 0.8);
////    std::cout << " before filter = " << cloud_brightness->size() <<
////                 " after filter " << filtered_cloud->size() << std::endl;
//    // Brightness filtered
//    std::string fName_br_fil = "./Results/brightness_filtered.pcd";
//    filtered_cloud->width = 1;
//    filtered_cloud->height = filtered_cloud->points.size();
//    pcl::io::savePCDFileASCII(fName_br_fil , *filtered_cloud);

    // Geometry
    std::string fName_geo = "./Results/geometry.pcd";
    cloud_geometry->width = 1;
    cloud_geometry->height = cloud_geometry->points.size();
    pcl::io::savePCDFileASCII(fName_geo , *cloud_geometry);


    // Global
    std::string fName_global = "./Results/global.pcd";
    br_regularizer_cloud_g->width = 1;
    br_regularizer_cloud_g->height = br_regularizer_cloud_g->points.size();
    pcl::io::savePCDFileASCII(fName_global , *br_regularizer_cloud_g);

    // Global1
    std::string fName_global1 = "./Results/global1.pcd";
    br_regularizer_cloud_g1->width = 1;
    br_regularizer_cloud_g1->height = br_regularizer_cloud_g1->points.size();
    pcl::io::savePCDFileASCII(fName_global1 , *br_regularizer_cloud_g1);

//    // Geometry
//    std::string fName_geo = "./Results/geometry.pcd";
//    cloud_geometry->width = 1;
//    cloud_geometry->height = cloud_geometry->points.size();
//    pcl::io::savePCDFileASCII(fName_geo , *cloud_geometry);
}
