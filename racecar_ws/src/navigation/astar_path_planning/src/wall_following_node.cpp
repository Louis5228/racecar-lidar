#include "wall_following.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "wall_following_node");
  ros::NodeHandle nh, pnh("~");
  WallFollower wf(nh, pnh);
  while(ros::ok()) ros::spinOnce();
  return 0;
}
