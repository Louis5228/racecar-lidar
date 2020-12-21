#include <iterator>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <gazebo_msgs/LinkStates.h>
#include <tf/transform_broadcaster.h>

const std::string target = "/::base_link";
ros::Publisher pub_odom;
nav_msgs::Odometry odom;

void cb(const gazebo_msgs::LinkStates msg)
{
  int idx;
  static tf::TransformBroadcaster br;
  tf::Transform transform;
  // Convert msg name to std::vector<std::string>
  std::vector<std::string> vec;
  for(auto it=msg.name.begin(); it!=msg.name.end(); ++it){
    std::string temp = *it;
    vec.push_back(temp);
  }
  std::vector<std::string>::iterator it;
  it = std::find(vec.begin(), vec.end(), target);
  if(it!=vec.end()){
    idx = std::distance(vec.begin(), it);
    odom.pose.pose.position.x = msg.pose[idx].position.x;
    odom.pose.pose.position.y = msg.pose[idx].position.y;
    odom.pose.pose.orientation.x = msg.pose[idx].orientation.x;
    odom.pose.pose.orientation.y = msg.pose[idx].orientation.y;
    odom.pose.pose.orientation.z = msg.pose[idx].orientation.z;
    odom.pose.pose.orientation.w = msg.pose[idx].orientation.w;
    odom.twist.twist.linear.x = msg.twist[idx].linear.x;
    odom.twist.twist.angular.z = msg.twist[idx].angular.z;
    pub_odom.publish(odom);
    transform.setOrigin(tf::Vector3(odom.pose.pose.position.x, odom.pose.pose.position.y, 0.0));
    tf::Quaternion q(odom.pose.pose.orientation.x, 
                     odom.pose.pose.orientation.y,
                     odom.pose.pose.orientation.z,
                     odom.pose.pose.orientation.w);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "odom", "base_link"));
  }
  else{ROS_WARN("Cannot find %s.", target.c_str());}
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "gazebo_groundtruth_node");
  ros::NodeHandle nh("~");
  // Initial odometry
  odom.header.frame_id = "odom";
  odom.pose.covariance[0] = 0.001;
  odom.pose.covariance[7] = 0.001;
  odom.pose.covariance[14] = 0.001;
  odom.pose.covariance[21] = 0.001;
  odom.pose.covariance[28] = 0.001;
  odom.pose.covariance[35] = 0.001;
  odom.twist.covariance[0] = 0.001;
  odom.twist.covariance[35] = 0.001;
  ros::Subscriber sub_link_state = nh.subscribe("/gazebo/link_states", 1, cb);
  pub_odom = nh.advertise<nav_msgs::Odometry>("fake_odom", 1);
  while(ros::ok()) ros::spinOnce();
}
