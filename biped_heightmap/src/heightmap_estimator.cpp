#include <ros/ros.h>
#include <jsk_pcl_ros/heightmap_utils.h>
#include <jsk_recognition_msgs/HeightmapConfig.h>
#include <jsk_topic_tools/diagnostic_nodelet.h>
#include <cv_bridge/cv_bridge.h>
#include <mutex>

namespace biped_heightmap {

  class HeightmapEstimator: public jsk_topic_tools::DiagnosticNodelet
  {
  public:
    typedef boost::shared_ptr<HeightmapEstimator> Ptr;
    HeightmapEstimator(): DiagnosticNodelet("HeightmapEstimator") {}

  protected:
    virtual void onInit();
    virtual void subscribe();
    virtual void unsubscribe();
    virtual void convert(const sensor_msgs::Image::ConstPtr& msg);
    virtual void configCallback(
      const jsk_recognition_msgs::HeightmapConfig::ConstPtr& msg);
    jsk_recognition_msgs::HeightmapConfig::ConstPtr config_msg_;
    std::mutex mutex_;
    ros::Publisher pub_;
    ros::Publisher pub_config_;
    ros::Subscriber sub_;
    ros::Subscriber sub_config_;
    double min_x_;
    double max_x_;
    double min_y_;
    double max_y_;
  };

  void HeightmapEstimator::onInit()
  {
    DiagnosticNodelet::onInit();
    pub_config_ = pnh_->advertise<jsk_recognition_msgs::HeightmapConfig>(
      "output/config", 1);
    sub_config_ = pnh_->subscribe(
      jsk_pcl_ros::getHeightmapConfigTopic(pnh_->resolveName("input")), 1,
      &HeightmapEstimator::configCallback, this);
    pub_ = advertise<sensor_msgs::Image>(*pnh_, "output", 1);
    onInitPostProcess();
  }

  void HeightmapEstimator::subscribe()
  {
    sub_ = pnh_->subscribe("input", 1, &HeightmapEstimator::convert, this);
  }

  void HeightmapEstimator::unsubscribe()
  {
    sub_.shutdown();
  }

  void HeightmapEstimator::configCallback(
    const jsk_recognition_msgs::HeightmapConfig::ConstPtr& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    config_msg_ = msg;
    min_x_ = msg->min_x;
    max_x_ = msg->max_x;
    min_y_ = msg->min_y;
    max_y_ = msg->max_y;
    pub_config_.publish(msg);
  }

  void HeightmapEstimator::convert(const sensor_msgs::Image::ConstPtr& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_msg_) {
      NODELET_ERROR("no ~input/config is yet available");
      return;
    }

    cv::Mat float_image = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC2)->image;
    ROS_WARN("b");
    int height = float_image.rows;
    int width  = float_image.cols;

    //inFiniteな部分には隣のzをコピー
    for (int j = 0; j < height; j++) {
      for (int i = 0; i < width; i++) {
        float v = float_image.at<cv::Vec2f>(j, i)[0];
        if (v == -FLT_MAX) {
          if (i > 0 && float_image.at<cv::Vec2f>(j, i-1)[0] != -FLT_MAX){
            float_image.at<cv::Vec2f>(j, i) = float_image.at<cv::Vec2f>(j, i-1);
          }else if (j > 0 && float_image.at<cv::Vec2f>(j-1, i)[0] != -FLT_MAX){
            float_image.at<cv::Vec2f>(j, i) = float_image.at<cv::Vec2f>(j-1, i);
          }
        }
      }
    }

    for (int j = height-1; j >= 0; j--) { // inverse order
      for (int i = width-1; i >= 0; i--) {
        float v = float_image.at<cv::Vec2f>(j, i)[0];
        if (v == -FLT_MAX) {
          if (i < width-1 && float_image.at<cv::Vec2f>(j, i+1)[0] != -FLT_MAX){
            float_image.at<cv::Vec2f>(j, i) = float_image.at<cv::Vec2f>(j, i+1);
          }else if (j < height-1 && float_image.at<cv::Vec2f>(j+1, i)[0] != -FLT_MAX){
            float_image.at<cv::Vec2f>(j, i) = float_image.at<cv::Vec2f>(j+1, i);
          }
        }
      }
    }

    // Convert to sensor_msgs/Image
    cv_bridge::CvImage height_map_image(msg->header,
                                        sensor_msgs::image_encodings::TYPE_32FC2,
                                        float_image);
    pub_.publish(height_map_image.toImageMsg());
  }

};

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS (biped_heightmap::HeightmapEstimator, nodelet::Nodelet);
