#!/usr/bin/env python

import sensor_msgs.point_cloud2 as pc2
import rospy
from sensor_msgs.msg import PointCloud2, LaserScan
import laser_geometry.laser_geometry as lg
import math

rospy.init_node("laserscan_to_pointcloud")

lp = lg.LaserProjection()

#pc_pub = rospy.Publisher("RL/VAE_mmwave_pc", PointCloud2, queue_size=1)
pc_pub = rospy.Publisher("/racecar/velodyne1/velodyne_points", PointCloud2, queue_size=1)

def scan_cb(msg):
	# convert the message of type LaserScan to a PointCloud2
	pc2_msg = lp.projectLaser(msg)

	# publish
	pc_pub.publish(pc2_msg)
	
	# convert it to a generator of the individual points
	point_generator = pc2.read_points(pc2_msg)

#rospy.Subscriber("RL/denoised_mmwave", LaserScan, scan_cb, queue_size=1)
rospy.Subscriber("/scan", LaserScan, scan_cb, queue_size=1)
rospy.spin()
