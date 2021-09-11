#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <ctime>
#include <iomanip>
#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/vfh.h>
#include <pcl/features/3dsc.h>
#include <stdio.h>

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "ceres/rotation.h"


using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

#define DIV_FACTOR			1.4		// Division factor used for graduated non-convexity
#define USE_ABSOLUTE_SCALE	0		// Measure distance in absolute scale (1) or in scale relative to the diameter of the model (0)
#define MAX_CORR_DIST		0.025	// Maximum correspondence distance (also see comment of USE_ABSOLUTE_SCALE)
#define ITERATION_NUMBER	64		// Maximum number of iteration
#define TUPLE_SCALE			0.95	// Similarity measure used for tuples of feature points.
#define TUPLE_MAX_CNT		1000	// Maximum tuple numbers.

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
//typedef pcl::ReferenceFrame RFType;
typedef pcl::FPFHSignature33 DescriptorType;

typedef pcl::visualization::PointCloudColorHandlerCustom<PointType> ColorHandlerT;


std::string model_filename_;
std::string scene_filename_;

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool use_cloud_resolution_ (false);
bool use_hough_ (true);
float model_ss_ (0.04f);
float scene_ss_ (0.04f);
//float rf_rad_ (model_ss_*5);
float descr_rad_ (model_ss_*3);
//float cg_thresh_ (5.0f);

void
showHelp (char *filename)
{
    std::cout << std::endl;
    std::cout << "***************************************************************************" << std::endl;
    std::cout << "*                                                                         *" << std::endl;
    std::cout << "*             Correspondence Grouping Tutorial - Usage Guide              *" << std::endl;
    std::cout << "*                                                                         *" << std::endl;
    std::cout << "***************************************************************************" << std::endl << std::endl;
    std::cout << "Usage: " << filename << " model_filename.pcd scene_filename.pcd [Options]" << std::endl << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "     -h:                     Show this help." << std::endl;
    std::cout << "     -k:                     Show used keypoints." << std::endl;
    std::cout << "     -c:                     Show used correspondences." << std::endl;
    std::cout << "     -r:                     Compute the model cloud resolution and multiply" << std::endl;
    std::cout << "                             each radius given by that value." << std::endl;
    std::cout << "     --algorithm (Hough|GC): Clustering algorithm used (default Hough)." << std::endl;
    std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;
    std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;
    std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
    std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
    std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
    std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
}


void
parseCommandLine (int argc, char *argv[])
{
    //Show help
    if (pcl::console::find_switch (argc, argv, "-h"))
    {
        showHelp (argv[0]);
        exit (0);
    }

    //Model & scene filenames
    std::vector<int> filenames;
    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".txt");
    if (filenames.size () != 2)
    {
        std::cout << "Filenames missing.\n";
        showHelp (argv[0]);
        exit (-1);
    }

    model_filename_ = argv[filenames[0]];
    scene_filename_ = argv[filenames[1]];

    //Program behavior
//    if (pcl::console::find_switch (argc, argv, "-k"))
//    {
        show_keypoints_ = true;
//    }
//    if (pcl::console::find_switch (argc, argv, "-c"))
//    {
        show_correspondences_ = true;
//    }
    if (pcl::console::find_switch (argc, argv, "-r"))
    {
        use_cloud_resolution_ = true;
    }

    std::string used_algorithm;
    if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
    {
        if (used_algorithm.compare ("Hough") == 0)
        {
            use_hough_ = true;
        }else if (used_algorithm.compare ("GC") == 0)
        {
            use_hough_ = false;
        }
        else
        {
            std::cout << "Wrong algorithm name.\n";
            showHelp (argv[0]);
            exit (-1);
        }
    }

    //General parameters
    pcl::console::parse_argument (argc, argv, "--model_ss", model_ss_);
    pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss_);
//    pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad_);
    pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad_);
//    pcl::console::parse_argument (argc, argv, "--cg_size", cg_size_);
//    pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh_);
}

double
computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
    double res = 0.0;
    int n_points = 0;
    int nres;
    std::vector<int> indices (2);
    std::vector<float> sqr_distances (2);
    pcl::search::KdTree<PointType> tree;
    tree.setInputCloud (cloud);

    for (std::size_t i = 0; i < cloud->size (); ++i)
    {
        if (! std::isfinite ((*cloud)[i].x))
        {
            continue;
        }
        //Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
        if (nres == 2)
        {
            res += sqrt (sqr_distances[1]);
            ++n_points;
        }
    }
    if (n_points != 0)
    {
        res /= n_points;
    }
    return res;
}

void CreateRandomValue(std::vector<int> &randomIdx, uint32_t rangeMax)
{
//    std::cout << std::endl;
    while(randomIdx.size()<1)
   {

        int value = rand()%rangeMax;
        int flag=1;
//        for(size_t i=0; i<randomIdx.size();++i){
//            if (randomIdx[i] == value){
//                flag=0;
//            }
//        }

        if(flag==1){
            randomIdx.push_back(value);
//            std::cout << ", " << value << endl;
        }

    }
}

void getCenter(pcl::PointCloud<PointType>::Ptr  &model_point, float &x, float &y, float &z){
    float x_min=10,y_min=10,z_min=10,x_max=0,y_max=0,z_max=0;
    for(size_t i =0; i<model_point->size();++i){
        if ( model_point->at(i).x<x_min)
            x_min  = model_point->at(i).x;
        if ( model_point->at(i).x>x_max)
            x_max  = model_point->at(i).x;

        if ( model_point->at(i).y<y_min)
            y_min  = model_point->at(i).y;
        if ( model_point->at(i).y>y_max)
            y_max  = model_point->at(i).y;

        if ( model_point->at(i).z<z_min)
            z_min  = model_point->at(i).z;
        if ( model_point->at(i).z>z_max)
            z_max  = model_point->at(i).z;
    }
    x = (x_max + x_min)/2;
    y = (y_max + y_min)/2;
    z = (z_max + z_min)/2;

}
float estimateScale(pcl::PointCloud<PointType>::Ptr  &model_point, pcl::PointCloud<PointType>::Ptr  &scene_point){

    float x1=0,y1=0,z1=0,x2=0,y2=0,z2=0;
//    getCenter(model_point, x1, y1, z1);
//    getCenter(scene_point, x2, y2, z2);



    for(size_t i =0; i<model_point->size();++i){
        x1 += model_point->at(i).x;
        y1 += model_point->at(i).y;
        z1 += model_point->at(i).z;

    }
    for(size_t i =0; i<scene_point->size();++i){
        x2 += scene_point->at(i).x;
        y2 += scene_point->at(i).y;
        z2 += scene_point->at(i).z;
    }

    float scale1=0, scale2=0;
    x1 /=model_point->size();
    y1 /=model_point->size();
    z1 /=model_point->size();

    x2 /=scene_point->size();
    y2 /=scene_point->size();
    z2 /=scene_point->size();

    Eigen::Vector3f model_distance;
    Eigen::Vector3f scene_distance;
    for (std::size_t i=0; i < model_point->size(); i++) {
        model_distance[0] = model_point->at(i).x - x1;
        model_distance[1] = model_point->at(i).y - y1;
        model_distance[2] = model_point->at(i).z - z1;
        scale1 += model_distance.squaredNorm() ;
    }

    for (std::size_t i=0; i < scene_point->size(); i++) {
        scene_distance[0] = scene_point->at(i).x - x2;
        scene_distance[1] = scene_point->at(i).y - y2;
        scene_distance[2] = scene_point->at(i).z - z2;
        scale2 += scene_distance.squaredNorm() ;

    }
    scale1 = sqrt(scale1) /model_point->size();
    scale2 = sqrt(scale2) /scene_point->size();
    return scale2/scale1;

}

void writeBinFile(std::string file_path,  pcl::PointCloud<PointType>::Ptr & object, pcl::PointCloud<DescriptorType>::Ptr &object_features){
    FILE* fid = fopen(file_path.c_str(), "wb");
    int nV = object->size(), nDim = 33;
    fwrite(&nV, sizeof(int), 1, fid);
    fwrite(&nDim, sizeof(int), 1, fid);
    for (int v = 0; v < nV; v++) {
        const PointType &pt = object->points[v];
        float xyz[3] = {pt.x, pt.y, pt.z};
        fwrite(xyz, sizeof(float), 3, fid);
        const pcl::FPFHSignature33 &feature = object_features->points[v];
        fwrite(feature.histogram, sizeof(float), 33, fid);
    }
    fclose(fid);
}



void optimizer(pcl::PointCloud<PointType>::Ptr  &model_point, pcl::PointCloud<PointType>::Ptr  &scene_point, Eigen::MatrixXf &params){
    ceres::Problem problem;}




struct RegistrationError {
    RegistrationError(PointType scene_pt, PointType match_pt, double weight)
            : scene_pt(scene_pt), match_pt(match_pt), weight(weight) {}

    template<typename T>
    bool operator()(const T *const trans_param, T *residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        T costheta = cos(trans_param[0]);
        T sintheta = sin(trans_param[0]);
        p[0] = costheta * (double)scene_pt.x - sintheta * (double)scene_pt.y + trans_param[1];
        p[1] = sintheta * (double)scene_pt.x + costheta * (double)scene_pt.y + trans_param[2];
        p[2] = (double)scene_pt.z + trans_param[3];

        residuals[0] = weight*(p[0] - (double)match_pt.x);
        residuals[1] = weight*(p[1] - (double)match_pt.y);
        residuals[2] = weight*(p[2] - (double)match_pt.z);

        return true;
    }

    static ceres::CostFunction* Create(PointType scene_pt, PointType match_pt, double weight) {
        return (new ceres::AutoDiffCostFunction<RegistrationError, 3, 4>(
                new RegistrationError(scene_pt, match_pt, weight)));
    }
    PointType scene_pt;
    PointType match_pt;
    double weight;

};

struct  SymmetryError{
    SymmetryError(PointType scene_pt, PointType match_pt, double weight)
    : scene_pt(scene_pt), match_pt(match_pt), weight(weight) {}

    template <typename T>
    bool operator()(const T *const sym_param, T *residuals) const{
        T p[3];
        T costheta = cos(sym_param[0]);
        T sintheta = sin(sym_param[0]);

        p[0] = (double)scene_pt.x - 2.0 * costheta * (costheta*(double)scene_pt.x + sintheta*(double)scene_pt.y + sym_param[1]);
        p[1] = (double)scene_pt.y - 2.0 * sintheta * (costheta*(double)scene_pt.x + sintheta*(double)scene_pt.y + sym_param[1]);
//        p[2] = (double)scene_pt.z;

        residuals[0] = weight * (p[0] - (double)match_pt.x);
        residuals[1] = weight * (p[1] - (double)match_pt.y);
//        residuals[2] = weight * ((double)scene_pt.z- (double)match_pt.z);

    }
    static ceres::CostFunction* Create(PointType scene_pt, PointType match_pt, double weight) {
        return (new ceres::AutoDiffCostFunction<SymmetryError, 2, 2>
                (new SymmetryError(scene_pt, match_pt, weight)));
    }
    PointType scene_pt;
    PointType match_pt;
    double weight;
};

struct Corr
{
    Corr(int s,int t): s(s),t(t){};
    int s;
    int t;
    int flag = 0;
};
template <typename T>
T estimate_trans_weight(PointType scene_pt, PointType match_pt,T *trans_param, float mu) {
    T p[3],residual ;
    T costheta = cos(trans_param[0]);
    T sintheta = sin(trans_param[0]);
    p[0] = costheta * scene_pt.x - sintheta * scene_pt.y + trans_param[1];
    p[1] = sintheta * scene_pt.x + costheta * scene_pt.y + trans_param[2];
    p[2] = scene_pt.z + trans_param[3];

    residual = sqrt( (p[0] - match_pt.x)*(p[0] - match_pt.x) + (p[1] - match_pt.y)*(p[1] - match_pt.y) + (p[2] - match_pt.z)*(p[2] - match_pt.z) );
    residual = mu/(mu+residual);
    return residual;
}
template <typename T>
T estimate_symm_weight(PointType scene_pt, PointType match_pt, T *sym_param, float mu){

    T p[3],residual;
    T costheta = cos(sym_param[0]);
    T sintheta = sin(sym_param[0]);

    p[0] = scene_pt.x - 2*costheta * (costheta*scene_pt.x + sintheta*scene_pt.y + sym_param[1]);
    p[1] = scene_pt.y - 2*sintheta * (costheta*scene_pt.x + sintheta*scene_pt.y + sym_param[1]);
    p[2] = scene_pt.z;

    residual = sqrt((p[0] - match_pt.x) *(p[0] - match_pt.x) +(p[1] - match_pt.y)*(p[1] - match_pt.y) + (p[2] - match_pt.z)*(p[2] - match_pt.z));
    residual = mu / (mu+residual);
    return residual;
}
bool readTxtFile(const  std::string fileName,  const char tag,  pcl::PointCloud<PointType>::Ptr &pointCloud)
{
    cout << "reading file start:  " << fileName<< endl;
    ifstream fin(fileName);
    if (!fin){
        std::cout<<"can not open the file "<< fileName << std::endl;
    };

    std::string linestr;
    std::vector<PointType> myPoint;
    getline(fin, linestr);

    while (getline(fin, linestr))
    {

        std::vector<std::string> strvec;
        std::string s;
        std::stringstream ss(linestr);
//        cout << "reading file start:  " << linestr << endl;

        while (getline(ss, s, tag))
        {
            if (s.size()!=0 && s[0]!=' '){
                strvec.push_back(s);
//                cout << "reading file start:  " << s << endl;
             }
//


        }
        if (strvec.size() < 3){
            cout << "格式不支持" << endl;
            return false;
        }
        PointType p;
        double scale=5.0;
        p.x = scale*stod(strvec[0]);
        p.y = scale*stod(strvec[1]);
        p.z = scale*stod(strvec[2]);
        myPoint.push_back(p);
    }
    fin.close();

    //转换成pcd
    pointCloud->width = (int)myPoint.size();
    pointCloud->height = 1;
    pointCloud->is_dense = false;
    pointCloud->points.resize(pointCloud->width * pointCloud->height);
    for (int i = 0; i < myPoint.size(); i++)
    {
        pointCloud->points[i].x = myPoint[i].x;
        pointCloud->points[i].y = myPoint[i].y;
        pointCloud->points[i].z = myPoint[i].z;
    }
    cout << "reading file finished! " << endl;
    cout << "There are " << pointCloud->points.size() << " points!" << endl;
    return true;
}

void AdvancedMatching(pcl::PointCloud<PointType>::Ptr &source_pt, pcl::PointCloud<PointType>::Ptr &target_pt,
                      pcl::PointCloud<DescriptorType>::Ptr &source_features, pcl::PointCloud<DescriptorType>::Ptr &target_features,
                      std::vector<std::pair<int, int> > &corres)
{


    int nPti = source_pt->size();
    int nPtj = target_pt->size();

    ///////////////////////////
    /// BUILD FLANNTREE
    ///////////////////////////

    pcl::KdTreeFLANN<DescriptorType> search_target;
    pcl::KdTreeFLANN<DescriptorType> search_source;

    search_target.setInputCloud (target_features);
    search_source.setInputCloud (source_features);



    bool crosscheck = true;
    bool tuple = true;

    std::vector<int> corres_K, corres_K2;
    std::vector<float> dis;
    std::vector<int> ind;

//    std::vector<std::pair<int, int> > corres;
    std::vector<std::pair<int, int> > corres_cross;
    std::vector<std::pair<int, int> > corres_ij;
    std::vector<std::pair<int, int> > corres_ji;

    ///////////////////////////
    /// INITIAL MATCHING
    ///////////////////////////

    std::vector<int> i_to_j(nPti, -1);
    for (int j = 0; j < nPtj; j++)
    {

        std::vector<int> neigh_indices (1), neigh_indices1 (1);
        std::vector<float> neigh_sqr_dists (1), neigh_sqr_dists1 (1);
        int found_neighs = search_target.nearestKSearch(source_features->at(j), 1, neigh_indices, neigh_sqr_dists);

        int i = neigh_indices[0];
        if (i_to_j[i] == -1)
        {
            int found_neighs = search_source.nearestKSearch(target_features->at(i), 1, neigh_indices1, neigh_sqr_dists1);

            int ij = neigh_indices1[0];
            i_to_j[i] = ij;
        }
        corres_ji.push_back(std::pair<int, int>(j,i));
    }



    for (int i = 0; i < nPti; i++)
    {
        if (i_to_j[i] != -1)
            corres_ij.push_back(std::pair<int, int>(i, i_to_j[i]));
    }

    int ncorres_ij = corres_ij.size();
    int ncorres_ji = corres_ji.size();

    // corres = corres_ij + corres_ji;
    for (int i = 0; i < ncorres_ij; ++i)
        corres.push_back(std::pair<int, int>(corres_ij[i].first, corres_ij[i].second));
    for (int j = 0; j < ncorres_ji; ++j)
        corres.push_back(std::pair<int, int>(corres_ji[j].first, corres_ji[j].second));

    printf("Number of points that remain: %d\n", (int)corres.size());

    ///////////////////////////
    /// CROSS CHECK
    /// input : corres_ij, corres_ji
    /// output : corres
    ///////////////////////////
    if (crosscheck)
    {
        printf("\t[cross check] ");

        // build data structure for cross check
        corres.clear();
        corres_cross.clear();
        std::vector<std::vector<int> > Mi(nPti);
        std::vector<std::vector<int> > Mj(nPtj);

        int ci, cj;
        for (int i = 0; i < ncorres_ij; ++i)
        {
            ci = corres_ij[i].first;
            cj = corres_ij[i].second;
            Mi[ci].push_back(cj);
        }
        for (int j = 0; j < ncorres_ji; ++j)
        {
            ci = corres_ji[j].first;
            cj = corres_ji[j].second;
            Mj[cj].push_back(ci);
        }


        // cross check
        for (int i = 0; i < nPti; ++i)
        {
            for (int ii = 0; ii < Mi[i].size(); ++ii)
            {
                int j = Mi[i][ii];
                for (int jj = 0; jj < Mj[j].size(); ++jj)
                {
                    if (Mj[j][jj] == i)
                    {
                        corres.push_back(std::pair<int, int>(i, j));
                        corres_cross.push_back(std::pair<int, int>(i, j));
                    }
                }
            }
        }
        printf("Number of points that remain after cross-check: %d\n", (int)corres.size());
    }

    ///////////////////////////
    /// TUPLE CONSTRAINT
    /// input : corres
    /// output : corres
    ///////////////////////////
    if (tuple)
    {
        srand(time(NULL));

        printf("\t[tuple constraint] ");
        int rand0, rand1, rand2;
        int idi0, idi1, idi2;
        int idj0, idj1, idj2;
        float scale = TUPLE_SCALE;
        int ncorr = corres.size();
        int number_of_trial = ncorr * 100;
        std::vector<std::pair<int, int> > corres_tuple;

        int cnt = 0;
        int i;
        for (i = 0; i < number_of_trial; i++)
        {
            rand0 = rand() % ncorr;
            rand1 = rand() % ncorr;
            rand2 = rand() % ncorr;

            idi0 = corres[rand0].first;
            idj0 = corres[rand0].second;
            idi1 = corres[rand1].first;
            idj1 = corres[rand1].second;
            idi2 = corres[rand2].first;
            idj2 = corres[rand2].second;

            // collect 3 points from i-th fragment
            Eigen::Vector3f pti0;
            pti0<< source_pt->at(idi0).x, source_pt->at(idi0).y, source_pt->at(idi0).z;
            Eigen::Vector3f pti1;
            pti1<< source_pt->at(idi1).x, source_pt->at(idi1).y, source_pt->at(idi1).z;
            Eigen::Vector3f pti2;
            pti2<< source_pt->at(idi2).x, source_pt->at(idi2).y, source_pt->at(idi2).z;

            float li0 = (pti0 - pti1).norm();
            float li1 = (pti1 - pti2).norm();
            float li2 = (pti2 - pti0).norm();

            // collect 3 points from j-th fragment
            Eigen::Vector3f ptj0;
            pti0 << target_pt->at(idj0).x, target_pt->at(idj0).y, target_pt->at(idj0).z;
            Eigen::Vector3f ptj1;
            ptj1 << target_pt->at(idj1).x, target_pt->at(idj1).y, target_pt->at(idj1).z;
            Eigen::Vector3f ptj2;
            ptj2 << target_pt->at(idj2).x, target_pt->at(idj2).y, target_pt->at(idj2).z;

            float lj0 = (ptj0 - ptj1).norm();
            float lj1 = (ptj1 - ptj2).norm();
            float lj2 = (ptj2 - ptj0).norm();

            if ((li0 * scale < lj0) && (lj0 < li0 / scale) &&
                (li1 * scale < lj1) && (lj1 < li1 / scale) &&
                (li2 * scale < lj2) && (lj2 < li2 / scale))
            {
                corres_tuple.push_back(std::pair<int, int>(idi0, idj0));
                corres_tuple.push_back(std::pair<int, int>(idi1, idj1));
                corres_tuple.push_back(std::pair<int, int>(idi2, idj2));
                cnt++;
            }

            if (cnt >= TUPLE_MAX_CNT)
                break;
        }

        printf("%d tuples (%d trial, %d actual).\n", cnt, number_of_trial, i);
        corres.clear();

        for (int i = 0; i < corres_tuple.size(); ++i)
            corres.push_back(std::pair<int, int>(corres_tuple[i].first, corres_tuple[i].second));
    }

    printf("\t[final] matches %d.\n", (int)corres.size());
}
void calculatePlane(std::vector<int> &randomIdx, std::vector<Corr> &symmetry_corrs, pcl::PointCloud<PointType>::Ptr & scene_pt, pcl::PointCloud<PointType>::Ptr & model_pt, Eigen::Vector3f &normal, float &d)
{
    Eigen::Vector3f pt1,pt2,t;
//    cout << "randomIdx size: "<<randomIdx.size()<<endl;
    Corr corr = symmetry_corrs[randomIdx[0]];
    pt1 << scene_pt->at(corr.s).x, scene_pt->at(corr.s).y, scene_pt->at(corr.s).z;
    Eigen::Vector3f pti1;
    pt2<< model_pt->at(corr.t).x, model_pt->at(corr.t).y, model_pt->at(corr.t).z;
//    cout << "randomIdx size: "<<randomIdx.size()<<endl;

    normal = (pt1-pt2).transpose();
    normal[2] = 0 ;
    normal /=normal.norm();
//    cout << "normal : "<<normal<<endl;


    t = (pt1+pt2)/2.0;
    d = -normal.dot(t);

}

int
main (int argc, char *argv[])
{
    parseCommandLine (argc, argv);

    pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr model_out (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
    pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
    pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
    pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());
    //
    //  Load clouds
    //

    std::cout << "loading model clou......" << std::endl;

    if (readTxtFile(model_filename_, ' ', model)< 0)
    {
        std::cout << "Error loading model cloud." << std::endl;
        showHelp (argv[0]);
        return (-1);
    }
    if (readTxtFile(scene_filename_, ' ', scene) < 0)
    {
        std::cout << "Error loading scene cloud." << std::endl;
        showHelp (argv[0]);
        return (-1);

    }


    pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
    norm_est.setKSearch (15);
    norm_est.setInputCloud (model);
    norm_est.compute (*model_normals);

    norm_est.setKSearch (15);

    norm_est.setInputCloud (scene);
    norm_est.compute (*scene_normals);

    pcl::UniformSampling<PointType> uniform_sampling;
    uniform_sampling.setInputCloud (model);
    uniform_sampling.setRadiusSearch (model_ss_);
    uniform_sampling.filter (*model_keypoints);
    std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

    uniform_sampling.setInputCloud (scene);
    uniform_sampling.setRadiusSearch (scene_ss_);
    uniform_sampling.filter (*scene_keypoints);
    std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;


    pcl::console::print_highlight ("Estimating features...\n");
    pcl::FPFHEstimationOMP<PointType, NormalType, DescriptorType> descr_est;



//    pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
    descr_est.setRadiusSearch (descr_rad_);

    descr_est.setInputCloud (model_keypoints);
    descr_est.setInputNormals (model_normals);
    descr_est.setSearchSurface (model);
    descr_est.compute (*model_descriptors);

    descr_est.setInputCloud(scene_keypoints);
    descr_est.setInputNormals(scene_normals);
    descr_est.setSearchSurface (scene);
    descr_est.compute (*scene_descriptors);
    std::vector<Corr> model_scene_corrs;
    std::vector<Corr> symmetry_corrs;
    clock_t start,end;

    start=clock();		//程序开始计时






    model_scene_corrs.clear();
    symmetry_corrs.clear();
    pcl::KdTreeFLANN<DescriptorType> match_search;
    pcl::KdTreeFLANN<DescriptorType> symmetry_search;

    match_search.setInputCloud(model_descriptors);
    symmetry_search.setInputCloud(scene_descriptors);

    pcl::console::print_highlight("finish estimating features...\n");




    //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
    for (int i = 0; i < scene_descriptors->size(); ++i) {
        std::vector<int> neigh_indices(1), neigh_indices1(3);
        std::vector<float> neigh_sqr_dists(1), neigh_sqr_dists1(3);

        int found_neighs = match_search.nearestKSearch(scene_descriptors->at(i), 1, neigh_indices, neigh_sqr_dists);

        symmetry_search.nearestKSearch(model_descriptors->at(neigh_indices[0]), 1, neigh_indices1, neigh_sqr_dists1);
        if ( i == neigh_indices1[0] && neigh_sqr_dists[0]<0.25f && neigh_sqr_dists1[0]<0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
        {
            if (abs(scene_keypoints->at(i).z - model_keypoints->at(neigh_indices[0]).z)<0.001){
                Corr corr(i, neigh_indices[0]);
                model_scene_corrs.push_back(corr);
            }

        }
        neigh_sqr_dists1.clear();
        neigh_sqr_dists1.clear();
        symmetry_search.nearestKSearch(scene_descriptors->at(i), 3, neigh_indices1, neigh_sqr_dists1);

        if (neigh_sqr_dists1[2] < 0.25f ) {

             PointType p = scene_keypoints->at(neigh_indices1[2]);
             Eigen::Vector3f pt1,pt2;
             pt1 << p.x, p.y, p.z;
             pt2 << scene_keypoints->at(i).x, scene_keypoints->at(i).y, scene_keypoints->at(i).z;
            float error = (pt1-pt2).norm();
            if (error >  model_ss_) {
                Corr corr(i, neigh_indices1[2]);
                corr.flag = 1;
                symmetry_corrs.push_back(corr);
            };

        }
    }

        std::cout << "Correspondences found:  " << model_scene_corrs.size() << "  symmetry correspondences found:  "
                  << symmetry_corrs.size() << std::endl;

//    end=clock();		//程序结束用时
//    double endtime=(double)(end-start)/CLOCKS_PER_SEC;
//    cout<<"Total time:"<<endtime<<endl;		//s为单位
//    cout<<"Total time:"<<endtime*1000<<"ms"<<endl;

    uint32_t  num_symmetry_pair = model_scene_corrs.size();
//    double trans_param[4] = {0}, sym_param[2] = {0}, weight = 1.0, mu=0.4;
    ceres::Problem problem;
    Eigen::Matrix4f transform;
    float best_error=10000000.0;
    srand((int)time(0));
    for (int t = 0; t<10000; ++t){
        Eigen::Vector3f normal;
        float d;
        std::vector<int> randomIdx;
//        srand((int)time(0));
        CreateRandomValue(randomIdx,  num_symmetry_pair-1);

        calculatePlane(randomIdx, model_scene_corrs, scene_keypoints, model_keypoints, normal, d);
        Eigen::Matrix4f transform_temp;
//        cout << "normal: " << normal.transpose() << "d:  " << d << endl;
        transform_temp << 1 - 2*normal[0]*normal[0], -2*normal[0]*normal[1], - 2*normal[0]*normal[2], -2*normal[0]*d,
                - 2*normal[1]*normal[0],  1-2*normal[1]*normal[1], - 2*normal[1]*normal[2], -2*normal[1]*d,
                - 2*normal[2]*normal[0],  -2*normal[2]*normal[1], 1- 2*normal[2]*normal[2], -2*normal[2]*d,
                0, 0, 0, 1;

//        cout << transform_temp << endl;

        pcl::PointCloud<PointType>::Ptr rotated_scene_keypoints (new pcl::PointCloud<PointType> ());
        pcl::transformPointCloud (*scene_keypoints, *rotated_scene_keypoints, transform_temp);
        float error=0;
        for (int i =0;i<num_symmetry_pair;++i){
            Eigen::Vector3f pt1,pt2;
//    cout << "randomIdx size: "<<randomIdx.size()<<endl;
            Corr corr = model_scene_corrs[i];
            pt1 << rotated_scene_keypoints->at(corr.s).x, rotated_scene_keypoints->at(corr.s).y, rotated_scene_keypoints->at(corr.s).z;
            pt2 << model_keypoints->at(corr.t).x, model_keypoints->at(corr.t).y, model_keypoints->at(corr.t).z;
            error += (pt1-pt2).norm();

        }
        error = error/num_symmetry_pair;

        if (error < best_error){
//            transform = transform_temp;
            best_error = error;
            cout <<"best_error: " << best_error << endl;
//            cout  << transform_temp << endl;

            transform << transform_temp;


        }


    }
    cout  << transform << endl;

////    for (size_t t=0; t<20; ++t){
////        for(size_t i=0; i<model_scene_corrs.size();++i){
////            Corr corr = model_scene_corrs[i];
////            if (t>0)
////                weight = estimate_trans_weight(scene_keypoints->at(corr.s), model_keypoints->at(corr.t),trans_param,mu);
////            else
////                weight = 1;
////            ceres::CostFunction* cost_function = RegistrationError::Create(scene_keypoints->at(corr.s), model_keypoints->at(corr.t), weight);
////            problem.AddResidualBlock(cost_function,
////                                     NULL /* squared loss */,
////                                     trans_param);
////        }
//
////        for(size_t i=0; i<symmetry_corrs.size();++i){
////            Corr corr = symmetry_corrs[i];
////            ceres::CostFunction* cost_function;
////
////            if (corr.flag==1) {
////                if (t>0)
////                    weight = estimate_symm_weight(scene_keypoints->at(corr.s), scene_keypoints->at(corr.t),sym_param,mu);
////                else
////                    weight = 1;
////
////                 cost_function =
////                        SymmetryError::Create(scene_keypoints->at(corr.s),
////                                              scene_keypoints->at(corr.t), weight);
////            }
////            else{
////                if (t>0)
////                    weight = estimate_symm_weight(scene_keypoints->at(corr.s), model_keypoints->at(corr.t),sym_param,mu);
////                 else
////                    weight = 1;
////                 cost_function =
////                        SymmetryError::Create(scene_keypoints->at(corr.s),
////                                              model_keypoints->at(corr.t), weight);
////            }
////
////
////            problem.AddResidualBlock(cost_function,
////                                     NULL /* squared loss */,
////                                     sym_param);
////        }
//
//        ceres::Solver::Options options;
//        options.linear_solver_type = ceres::DENSE_QR;
////        options.minimizer_progress_to_stdout = true;
//
//        ceres::Solver::Summary summary;
//        ceres::Solve(options, &problem, &summary);
////        std::cout << summary.FullReport() << "\n";
//
//    }
//    std::cout << summary.FullReport() << "\n";








    // Perform alignment

//    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity(4,4);


//    transform(0,0) = cos(trans_param[0]);
//    transform(0,1) = -sin(trans_param[0]);
//    transform(1,0) = sin(trans_param[0]);
//    transform(1,1) = cos(trans_param[0]);
//    transform(0,3)=trans_param[1];
//    transform(1,3)=trans_param[2];
//    transform(2,3)=trans_param[3];

    pcl::console::print_highlight("showing...\n");

    pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
    pcl::visualization::PointCloudColorHandlerCustom<PointType> model_color_handler (model, 255, 0, 0);


    viewer.addPointCloud (model, model_color_handler, "model_cloud");

    pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

    if (show_correspondences_ || show_keypoints_)
    {
        //  We are translating the model so that it doesn't end in the middle of the scene representation
        pcl::transformPointCloud (*scene, *off_scene_model, Eigen::Vector3f (-0.2,0,0), Eigen::Quaternionf (1, 0, 0, 0));
        pcl::transformPointCloud (*scene_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-0.2,0,0), Eigen::Quaternionf (1, 0, 0, 0));

        pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
        viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
    }

//
//        pcl::PointCloud<PointType>::Ptr rotated_scene (new pcl::PointCloud<PointType> ());
//        pcl::transformPointCloud (*scene, *rotated_scene, transform);
//
//
//
//        pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_scene, 255, 255, 0);
//        viewer.addPointCloud (rotated_scene, rotated_model_color_handler, "transformed  scene");

    if (show_correspondences_)
    {
        for (std::size_t j = 0; j < model_scene_corrs.size (); j = j + 5)
        {
            std::stringstream ss_line;
            ss_line << "correspondence_line" << "_" << j;
            PointType& scene_point = off_scene_model_keypoints->at (model_scene_corrs[j].s);
            PointType& model_point = model_keypoints->at (model_scene_corrs[j].t);

            //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
            viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
        }

//        for (std::size_t j = 0; j < symmetry_corrs.size (); j = j + 5)
//        {
//            std::stringstream ss_line;
//            ss_line << "correspondence_line" << "_" << j;
//            PointType& scene_point = off_scene_model_keypoints->at (symmetry_corrs[j].s);
//            PointType& model_point = scene_keypoints->at (symmetry_corrs[j].t);
//
//            //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
//            viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
//        }


    }




    while (!viewer.wasStopped ())
    {
        viewer.spinOnce ();
    }

    return (0);
}