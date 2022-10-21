#include <ros/ros.h>
#include <jsk_pcl_ros/heightmap_utils.h>
#include <jsk_recognition_msgs/HeightmapConfig.h>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <jsk_topic_tools/diagnostic_nodelet.h>
#include <cv_bridge/cv_bridge.h>
#include <mutex>
#include <biped_heightmap_msgs/SteppableRegion.h>
#include <biped_heightmap_msgs/LandingPosition.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include "polypartition.h"

namespace biped_heightmap {

  class SteppableRegionPublisher: public jsk_topic_tools::DiagnosticNodelet
  {
  public:
    typedef boost::shared_ptr<SteppableRegionPublisher> Ptr;
    SteppableRegionPublisher(): DiagnosticNodelet("SteppableRegionPublisher") {}

  protected:
    virtual void onInit();
    virtual void subscribe(){}
    virtual void unsubscribe(){}
    virtual void heightmapCallback(const sensor_msgs::Image::ConstPtr& msg);
    virtual void configCallback(
      const jsk_recognition_msgs::HeightmapConfig::ConstPtr& msg);
    virtual void targetCallback(const biped_heightmap_msgs::LandingPosition::ConstPtr& msg);
    jsk_recognition_msgs::HeightmapConfig::ConstPtr config_msg_;
    std::mutex mutex_;
    std::mutex mutex_target_;
    ros::Publisher pub_steppable_region_;
    ros::Publisher pub_visualized_image_;
    ros::Publisher pub_visualized_steppable_region_;
    ros::Subscriber sub_;
    ros::Subscriber sub_config_;
    ros::Subscriber sub_landing_target_;
    tf::TransformListener listener_;

    // heightmap config
    double min_x_;
    double max_x_;
    double min_y_;
    double max_y_;

    // param
    float dilate_range_; // [m]. radius
    float median_range_; // [m]. radius
    float obstacle_range_; // [m]
    float obstacle_height_; // [m]
    float steppable_range_; // [m]. steppable_range <= obstacle_range
    float steppable_terrain_angle_;// [rad]
    float steppable_slope_angle_;// [rad]
    float opening_range_; // [m]. radius
    float closing_range_; // [m]. radius
    std::string world_frame_; // heightmapをこのframeで保存する. heightmapが移動する座標系で表現されていてもよいように

    sensor_msgs::Image::ConstPtr heightmap_msg_;
    Eigen::Affine3f heightmap_pos_ = Eigen::Affine3f::Identity();
    cv::Mat median_image_;
    jsk_recognition_msgs::PolygonArray combined_meshes_;

    static bool compare_eigen3f(const Eigen::Vector3f& lv, const Eigen::Vector3f& rv) {
      return (lv(0) < rv(0)) || (lv(0) == rv(0) && lv(1) < rv(1)) || (lv(0) == rv(0) && lv(1) == rv(1) && lv(2) < rv(2));
    }

    static double calc_cross_product(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& o)  {
      return (a(0) - o(0)) * (b(1) - o(1)) - (a(1) - o(1)) * (b(0) - o(0));
    }

    static void calc_convex_hull (std::vector<Eigen::Vector3f>& vs, std::vector<Eigen::Vector3f>& ch) {
      int n_vs = vs.size(), n_ch = 0;
      ch.resize(2*n_vs);
      std::sort(vs.begin(), vs.end(), compare_eigen3f);
      for (int i = 0; i < n_vs; ch[n_ch++] = vs[i++])
        while (n_ch >= 2 && calc_cross_product(ch[n_ch-1], vs[i], ch[n_ch-2]) <= 0) n_ch--;
      for (int i = n_vs-2, j = n_ch+1; i >= 0; ch[n_ch++] = vs[i--])
        while (n_ch >= j && calc_cross_product(ch[n_ch-1], vs[i], ch[n_ch-2]) <= 0) n_ch--;
      ch.resize(n_ch-1);
    }

    static Eigen::Matrix3f orientCoordToAxis(const Eigen::Matrix3f& m, const Eigen::Vector3f& axis, const Eigen::Vector3f& localaxis = Eigen::Vector3f::UnitZ()){
      // axisとlocalaxisはノルムが1, mは回転行列でなければならない.
      // axisとlocalaxisがピッタリ180反対向きの場合、回転方向が定まらないので不安定
      Eigen::AngleAxisf m_ = Eigen::AngleAxisf(m); // Eigen::Matrix3dの空間で積算していると数値誤差によってだんたん回転行列ではなくなってくるので
      Eigen::Vector3f localaxisdir = m_ * localaxis;
      Eigen::Vector3f cross = localaxisdir.cross(axis);
      float dot = std::min(1.0f,std::max(-1.0f,localaxisdir.dot(axis))); // acosは定義域外のときnanを返す
      if(cross.norm()==0){
        if(dot == -1) return Eigen::Matrix3f(-m);
        else return Eigen::Matrix3f(m_);
      }else{
        float angle = std::acos(dot); // 0~pi
        Eigen::Vector3f axis = cross.normalized(); // include sign
        return Eigen::Matrix3f(Eigen::AngleAxisf(angle, axis) * m_);
      }
    }
  };

  void SteppableRegionPublisher::onInit()
  {
    DiagnosticNodelet::onInit();
    sub_ = pnh_->subscribe("input", 1, &SteppableRegionPublisher::heightmapCallback, this);
    sub_config_ = pnh_->subscribe(
      jsk_pcl_ros::getHeightmapConfigTopic(pnh_->resolveName("input")), 1,
      &SteppableRegionPublisher::configCallback, this);
    sub_landing_target_ = pnh_->subscribe("landing_target", 1, &SteppableRegionPublisher::targetCallback, this);
    pub_steppable_region_ = pnh_->advertise<biped_heightmap_msgs::SteppableRegion>("steppable_region", 1);
    pub_visualized_image_ = pnh_->advertise<sensor_msgs::Image> ("visualized_image", 1);
    pub_visualized_steppable_region_ = pnh_->advertise<jsk_recognition_msgs::PolygonArray> ("visualized_steppable_region", 1);
    pnh_->param<float>("dilate_range_", dilate_range_, 0.03);
    pnh_->param<float>("median_range_", median_range_, 0.02);
    pnh_->param<float>("obstacle_range", obstacle_range_, 0.15);
    pnh_->param<float>("obstacle_height", obstacle_height_, 0.04);
    pnh_->param<float>("steppable_range", steppable_range_, 0.12);
    pnh_->param<float>("steppable_terrain_angle", steppable_terrain_angle_, 0.20);
    pnh_->param<float>("steppable_slope_angle", steppable_slope_angle_, 0.35);
    pnh_->param<float>("opening_range_", opening_range_, 0.02);
    pnh_->param<float>("closing_range_", closing_range_, 0.02);
    pnh_->param<std::string>("world_frame_", world_frame_, std::string("odom"));

    steppable_range_ = std::min(steppable_range_, obstacle_range_);
    onInitPostProcess();
  }

  void SteppableRegionPublisher::configCallback(
    const jsk_recognition_msgs::HeightmapConfig::ConstPtr& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    config_msg_ = msg;
    min_x_ = msg->min_x;
    max_x_ = msg->max_x;
    min_y_ = msg->min_y;
    max_y_ = msg->max_y;
  }

  void SteppableRegionPublisher::targetCallback(const biped_heightmap_msgs::LandingPosition::ConstPtr& msg){
    std::lock_guard<std::mutex> lock(mutex_target_);

    if (!heightmap_msg_) {
      NODELET_ERROR("no ~input is yet available");
      return;
    }

    std::string target_frame;
    if (msg->l_r) {
      target_frame = "lleg_end_coords";
    }
    else {
      target_frame = "rleg_end_coords";
    }

    tf::StampedTransform transform;
    listener_.waitForTransform(world_frame_, target_frame, ros::Time(0)/*msg->header.stamp*/, ros::Duration(3.0));
    listener_.lookupTransform(world_frame_, target_frame, ros::Time(0)/*msg->header.stamp*/, transform); // map relative to target_frame

    Eigen::Affine3d cur_foot_pos_d;
    tf::transformTFToEigen(transform, cur_foot_pos_d);
    Eigen::Affine3f cur_foot_pos = cur_foot_pos_d.cast<float>();
    cur_foot_pos.linear() = orientCoordToAxis(cur_foot_pos.linear(), Eigen::Vector3f::UnitZ());
    Eigen::Affine3f cur_foot_pos_to_heightmap_pos = cur_foot_pos.inverse() * heightmap_pos_;

    biped_heightmap_msgs::SteppableRegion sr;
    sr.header.frame_id = target_frame;
    sr.header.stamp = msg->header.stamp;
    sr.l_r = msg->l_r;

    // convert to polygon relative to leg_end_coords
    sr.polygons.resize(combined_meshes_.polygons.size());
    for (size_t i = 0; i < combined_meshes_.polygons.size(); i++) {
      size_t vs_num(combined_meshes_.polygons[i].polygon.points.size());
      sr.polygons[i].header = sr.header;
      sr.polygons[i].polygon.points.resize(vs_num);
      for (size_t j = 0; j < vs_num; j++) {
        Eigen::Vector3f p_map(combined_meshes_.polygons[i].polygon.points[j].x, combined_meshes_.polygons[i].polygon.points[j].y, combined_meshes_.polygons[i].polygon.points[j].z);
        Eigen::Vector3f p = cur_foot_pos_to_heightmap_pos * p_map;
        sr.polygons[i].polygon.points[j].x = p[0];
        sr.polygons[i].polygon.points[j].y = p[1];
        sr.polygons[i].polygon.points[j].z = p[2];
      }
    }

    pub_steppable_region_.publish(sr);
  }

  void SteppableRegionPublisher::heightmapCallback(const sensor_msgs::Image::ConstPtr& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_msg_) {
      NODELET_ERROR("no ~input/config is yet available");
      return;
    }

    cv::Mat float_image = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC2)->image;

    int height = float_image.rows;
    int width  = float_image.cols;
    double dx = (max_x_ - min_x_) / width;
    double dy = (max_y_ - min_y_) / height;

    cv::Mat dilate_image = cv::Mat::zeros(height, width, CV_32FC2);
    cv::dilate(float_image, dilate_image, cv::noArray(), cv::Point(-1, -1), std::max(dilate_range_/dx, dilate_range_/dy)); // heightmapの中でまばらに欠落している点は-FLT_MAXが入っているので、埋める.
    cv::Mat median_image = cv::Mat::zeros(height, width, CV_32FC2);
    cv::medianBlur(dilate_image, median_image, 1 + std::max(median_range_/dx, median_range_/dy)); //中央値を取る(x,y座標はkernelの中心)

    cv::Mat binarized_image = cv::Mat::zeros(height, width, CV_8UC1); // 0: not steppable
    cv::Mat visualized_image = cv::Mat::zeros(height, width, CV_8UC3);

    int steppable_range_x = (int)(steppable_range_/dx);
    int steppable_range_y = (int)(steppable_range_/dy);
    float steppable_terrain_edge_height = steppable_range_*std::tan(steppable_terrain_angle_);
    float steppable_terrain_corner_height = steppable_terrain_edge_height*std::sqrt(2);
    float steppable_slope_edge_height = steppable_range_*std::tan(steppable_slope_angle_);
    float steppable_slope_corner_height = steppable_slope_edge_height*std::sqrt(2);

    int obstacle_edge_range_x = (int)(obstacle_range_/dx);
    int obstacle_edge_range_y = (int)(obstacle_range_/dy);
    int obstacle_corner_range_x = (int)(obstacle_range_/std::sqrt(2)/dx);
    int obstacle_corner_range_y = (int)(obstacle_range_/std::sqrt(2)/dy);

    float front = 0;
    float right = 0;
    float left = 0;
    float back = 0;
    bool front_flag = 0;
    bool right_flag = 0;
    bool left_flag = 0;
    bool back_flag= 0;

    for (int x = (int)(obstacle_edge_range_x); x < (median_image.cols-(int)(obstacle_edge_range_x)); x++) {
      for (int y = (int)(obstacle_edge_range_y); y < (median_image.rows-(int)(obstacle_edge_range_y)); y++) {
        //floor not exists
        if(dilate_image.at<cv::Vec2f>(y, x)[0] == -FLT_MAX){
          continue;
        }

        float center = median_image.at<cv::Vec2f>(y, x)[0];

        bool safe = true;
        for(int i = -steppable_range_x; i <= steppable_range_x; i++){
          for(int j = -steppable_range_y; j <= steppable_range_y; j++){
            float z = median_image.at<cv::Vec2f>(y+j, x+i)[0];
            if(z > center + steppable_terrain_edge_height || z < center - steppable_terrain_edge_height) safe = false;
          }
        }
        if(!safe) continue;

        // front = (median_image.at<cv::Vec2f>(y-steppable_range_y, x-steppable_range_x)[0] + median_image.at<cv::Vec2f>(y-steppable_range_y, x+steppable_range_x)[0]) / 2.0;
        // front_flag = true;
        // if (front + steppable_terrain_edge_height < median_image.at<cv::Vec2f>(y-steppable_range_y, x)[0]) {
        //   front = median_image.at<cv::Vec2f>(y-steppable_range_y, x)[0];
        //   front_flag = false;
        // }
        // right = (median_image.at<cv::Vec2f>(y-steppable_range_y, x+steppable_range_x)[0] + median_image.at<cv::Vec2f>(y+steppable_range_y, x+steppable_range_x)[0]) / 2.0;
        // right_flag = true;
        // if (right + steppable_terrain_edge_height < median_image.at<cv::Vec2f>(y, x+steppable_range_x)[0]) {
        //   right = median_image.at<cv::Vec2f>(y, x+steppable_range_x)[0];
        //   right_flag = false;
        // }
        // left = (median_image.at<cv::Vec2f>(y-steppable_range_y, x-steppable_range_x)[0] + median_image.at<cv::Vec2f>(y+steppable_range_y, x-steppable_range_x)[0]) / 2.0;
        // left_flag = true;
        // if (left + steppable_terrain_edge_height < median_image.at<cv::Vec2f>(y, x-steppable_range_x)[0]) {
        //   left = median_image.at<cv::Vec2f>(y, x-steppable_range_x)[0];
        //   left_flag = false;
        // }
        // back = (median_image.at<cv::Vec2f>(y+steppable_range_y, x-steppable_range_x)[0] + median_image.at<cv::Vec2f>(y+steppable_range_y, x+steppable_range_x)[0]) / 2.0;
        // back_flag = true;
        // if (back + steppable_terrain_edge_height < median_image.at<cv::Vec2f>(y+steppable_range_y, x)[0]) {
        //   back = median_image.at<cv::Vec2f>(y+steppable_range_y, x)[0];
        //   back_flag = false;
        // }

        // //凸性
        // if (std::abs(front + back - right - left) > steppable_terrain_edge_height*2) {
        //   continue;
        // }
        // center = (front + back + right + left) / 4.0;
        // if (center + steppable_terrain_edge_height < median_image.at<cv::Vec2f>(y, x)[0]) {
        //   continue;
        // }

        // //ひねり
        // if (front_flag && std::abs(median_image.at<cv::Vec2f>(y-steppable_range_y, x-steppable_range_x)[0] - median_image.at<cv::Vec2f>(y-steppable_range_y, x+steppable_range_x)[0] + right - left) > steppable_terrain_edge_height) {
        //   continue;
        // }
        // if (right_flag && std::abs(median_image.at<cv::Vec2f>(y-steppable_range_y, x+steppable_range_x)[0] - median_image.at<cv::Vec2f>(y+steppable_range_y, x+steppable_range_x)[0] + back - front) > steppable_terrain_edge_height) {
        //   continue;
        // }
        // if (left_flag && std::abs(median_image.at<cv::Vec2f>(y-steppable_range_y, x-steppable_range_x)[0] - median_image.at<cv::Vec2f>(y+steppable_range_y, x-steppable_range_x)[0] + back - front) > steppable_terrain_edge_height) {
        //   continue;
        // }
        // if (back_flag && std::abs(median_image.at<cv::Vec2f>(y+steppable_range_y, x-steppable_range_x)[0] - median_image.at<cv::Vec2f>(y+steppable_range_y, x+steppable_range_x)[0] + right - left) > steppable_terrain_edge_height) {
        //   continue;
        // }

        // //傾き
        // if (std::abs(front - back) > steppable_slope_edge_height*2) {
        //   continue;
        // }
        // if (std::abs(right - left) > steppable_slope_edge_height*2) {
        //   continue;
        // }

        visualized_image.at<cv::Vec3b>(y, x)[0] = 100;
        visualized_image.at<cv::Vec3b>(y, x)[1] = 100;
        visualized_image.at<cv::Vec3b>(y, x)[2] = 100;

        // obstacle
        if (
            median_image.at<cv::Vec2f>(y+obstacle_edge_range_y, x)[0] - center > obstacle_height_ ||
            median_image.at<cv::Vec2f>(y, x+obstacle_edge_range_x)[0] - center > obstacle_height_ ||
            median_image.at<cv::Vec2f>(y-obstacle_edge_range_y, x)[0] - center > obstacle_height_ ||
            median_image.at<cv::Vec2f>(y, x-obstacle_edge_range_x)[0] - center > obstacle_height_ ||
            median_image.at<cv::Vec2f>(y+obstacle_corner_range_y, x+obstacle_corner_range_x)[0] - center > obstacle_height_ ||
            median_image.at<cv::Vec2f>(y+obstacle_corner_range_y, x-obstacle_corner_range_x)[0] - center > obstacle_height_ ||
            median_image.at<cv::Vec2f>(y-obstacle_corner_range_y, x+obstacle_corner_range_x)[0] - center > obstacle_height_ ||
            median_image.at<cv::Vec2f>(y-obstacle_corner_range_y, x-obstacle_corner_range_x)[0] - center > obstacle_height_) {
          continue;
        }
        binarized_image.at<uchar>(y, x) = 255;
        visualized_image.at<cv::Vec3b>(y, x)[0] = 200;
        visualized_image.at<cv::Vec3b>(y, x)[1] = 200;
        visualized_image.at<cv::Vec3b>(y, x)[2] = 200;
      }
    }

    cv::morphologyEx(binarized_image, binarized_image, CV_MOP_CLOSE, cv::noArray(), cv::Point(-1, -1), std::max(opening_range_/dx, opening_range_/dy));
    cv::morphologyEx(binarized_image, binarized_image, CV_MOP_OPEN,  cv::noArray(), cv::Point(-1, -1), std::max(closing_range_/dx, closing_range_/dy));
    //cv::erode(binarized_image, binarized_image, cv::noArray(), cv::Point(-1, -1), erode_range);//3
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binarized_image, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

    std::vector<std::vector<cv::Point>> approx_vector;
    std::list<TPPLPoly> polys, result;
    for (int j = 0; j < contours.size(); j++) {
      if (hierarchy[j][3] == -1) { //外側
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[j], approx, 0.9, true);//1.5
        if (approx.size() >= 3) {
          approx_vector.push_back(approx);
          TPPLPoly poly;
          poly.Init(approx.size());
          for (int k = 0; k < approx.size(); k++) {
            poly[k].x = approx[k].x;
            poly[k].y = -approx[k].y;
          }
          polys.push_back(poly);
        }
      } else { //穴
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[j], approx, 1.0, true);
        if (approx.size() >= 3) {
          approx_vector.push_back(approx);
          TPPLPoly poly;
          poly.Init(approx.size());
          for (int k = 0; k < approx.size(); k++) {
            poly[k].x = approx[k].x;
            poly[k].y = -approx[k].y;
          }
          poly.SetHole(true);
          polys.push_back(poly);
        }
      }
    }
    TPPLPartition pp;
    pp.Triangulate_EC(&polys, &result);
    cv::drawContours(visualized_image, approx_vector, -1, cv::Scalar(255, 0, 0));

    std::vector<std::vector<Eigen::Vector3f> > meshes;

    for (std::list<TPPLPoly>::iterator iter = result.begin(); iter != result.end(); iter++) {
      std::vector<Eigen::Vector3f> mesh;
      //for (int j = 0; j < iter->GetNumPoints(); j++) {
      for (int j = iter->GetNumPoints() - 1; j >= 0; j--) {
        visualized_image.at<cv::Vec3b>(-iter->GetPoint(j).y, iter->GetPoint(j).x)[2] = 255;
        int p1 = msg->width * (-iter->GetPoint(j).y) + (iter->GetPoint(j).x);
        Eigen::Vector3f p;
        p[0] = iter->GetPoint(j).x * dx + min_x_ + dx / 2.0;
        p[1] = -iter->GetPoint(j).y * dy + min_y_ + dy / 2.0;
        p[2] = median_image.at<cv::Vec2f>(-iter->GetPoint(j).y, iter->GetPoint(j).x)[0];
        mesh.push_back(p);
      }
      meshes.push_back(mesh);
    }

    pub_visualized_image_.publish(cv_bridge::CvImage(msg->header, "bgr8", visualized_image).toImageMsg());

    std::vector<std::vector<Eigen::Vector3f> > combined_meshes;
    std::vector<std::vector<size_t> > combined_indices;
    std::vector<bool> is_combined(meshes.size(), false);
    for (size_t i = 0; i < meshes.size(); i++) {
      std::vector<size_t> is_combined_indices;
      is_combined_indices.push_back(i);
      for (size_t j = i + 1; j < meshes.size(); j++) {
        std::vector<Eigen::Vector3f> inter_v;
        std::vector<Eigen::Vector3f> v1 = meshes[i], v2 = meshes[j];
        std::sort(v1.begin(), v1.end(), compare_eigen3f);
        std::sort(v2.begin(), v2.end(), compare_eigen3f);
        std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), inserter(inter_v, inter_v.end()), compare_eigen3f);
        if (inter_v.size() == 2) { // adjacent mesh
          std::vector<Eigen::Vector3f> tmp_vs(v1), target_v, tmp_convex;
          std::set_difference(v2.begin(), v2.end(), v1.begin(), v1.end(), inserter(target_v, target_v.end()), compare_eigen3f);
          std::copy(target_v.begin(), target_v.end(), std::back_inserter(tmp_vs));
          calc_convex_hull(tmp_vs, tmp_convex);
          if (tmp_vs.size() == tmp_convex.size()) {
            meshes[i] = tmp_convex;
            meshes[j] = tmp_convex;
            is_combined[j] = true;
            is_combined_indices.push_back(j);
          }
        }
      }
      if (!is_combined[i]) {
        combined_meshes.push_back(meshes[i]);
        combined_indices.push_back(is_combined_indices);
      } else if (is_combined_indices.size() > 1) {
        for (size_t j = 0; j < combined_indices.size(); j++) {
          if (std::find(combined_indices[j].begin(), combined_indices[j].end(), i) != combined_indices[j].end()) {
            combined_meshes[j] = meshes[i];
            combined_indices[j] = is_combined_indices;
          }
        }
      }
      is_combined[i] = true;
    }

    tf::StampedTransform transform;
    listener_.waitForTransform(world_frame_, msg->header.frame_id, msg->header.stamp, ros::Duration(3.0));
    listener_.lookupTransform(world_frame_, msg->header.frame_id, msg->header.stamp, transform);
    Eigen::Affine3d heightmap_pos_d;
    tf::transformTFToEigen(transform, heightmap_pos_d);

    {
      std::lock_guard<std::mutex> lock(mutex_target_);

      heightmap_msg_ = msg;

      combined_meshes_.header = msg->header;
      combined_meshes_.polygons.resize(combined_meshes.size());
      for (size_t i = 0; i < combined_meshes.size(); i++) {
        size_t vs_num(combined_meshes[i].size());
        combined_meshes_.polygons[i].header = msg->header;
        combined_meshes_.polygons[i].polygon.points.resize(vs_num);
        for (size_t j = 0; j < vs_num; j++) {
          combined_meshes_.polygons[i].polygon.points[j].x = combined_meshes[i][j][0];
          combined_meshes_.polygons[i].polygon.points[j].y = combined_meshes[i][j][1];
          combined_meshes_.polygons[i].polygon.points[j].z = combined_meshes[i][j][2];
        }
      }
      pub_visualized_steppable_region_.publish(combined_meshes_);

      median_image_ = median_image;

      heightmap_pos_ = heightmap_pos_d.cast<float>();
    }
  }

};

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS (biped_heightmap::SteppableRegionPublisher, nodelet::Nodelet);
