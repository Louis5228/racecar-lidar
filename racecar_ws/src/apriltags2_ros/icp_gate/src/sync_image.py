#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo


class SyncImage(object):
    def __init__(self):
        self.info = None

        self.pub_info = rospy.Publisher('camera/color/camera_info',CameraInfo,queue_size=1)
        self.pub_image = rospy.Publisher('camera/color/image_raw',Image,queue_size=1)

        sub_info = rospy.Subscriber(
            '/husky2/camera_middle/color/camera_info', CameraInfo, self.cb_info, queue_size=1)
        sub_image = rospy.Subscriber(
            '/husky2/camera_middle/color/image_raw', Image, self.cb_image, queue_size=1)

    def cb_info(self, msg):
        self.info = msg

    def cb_image(self, msg):
        if self.info is not None:
            self.info.header.stamp = msg.header.stamp
            self.pub_image.publish(msg)
            self.pub_info.publish(self.info)

if __name__ == "__main__":
    rospy.init_node('resync')
    syncer = SyncImage()
    rospy.spin()
