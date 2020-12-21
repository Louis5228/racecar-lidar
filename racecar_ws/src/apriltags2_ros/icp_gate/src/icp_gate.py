#! /usr/bin/env python

import rospy
import tf
import numpy as np
from apriltags2_ros.msg import AprilTagDetectionArray, AprilTagDetection
from sklearn.svm import OneClassSVM

MAX_DETECT = 35


class ICP(object):
    def __init__(self):
        self.gate_poses = np.array([[0.501, 1.326, 0.596],      # tag 4
                                    [0.4675, 0.2515, 2.686],    # tag 5
                                    [0.4835, -1.34, 1.092]])    # tag 6
        self.iteration = 1
        self.count = 0
        self.listener = tf.TransformListener()
        self.broadcaster = tf.TransformBroadcaster()
        self.tiemr = None

        self.translations = np.zeros((MAX_DETECT, 3))
        self.orientations = np.zeros((MAX_DETECT, 4))
        self.trans_mean = np.zeros(3)
        self.orien_mean = np.zeros(4)

        # map/base_footprint 2 camera_color_optical_frame
        trans_c = [0.506, 0.033, 0.425]
        rot_c = [-0.430, 0.430, -0.561, 0.561]
        self.camera2foot = tf.transformations.concatenate_matrices(
            tf.transformations.translation_matrix(trans_c), tf.transformations.quaternion_matrix(rot_c))

        self.sub_detection = rospy.Subscriber(
            "tag_detections", AprilTagDetectionArray, self.cb_detection, queue_size=1)

    def pub_tf(self, event):
        # broadcast transformation
        self.broadcaster.sendTransform(
            self.trans_mean, self.orien_mean, rospy.Time.now(), "map", "global")
        # rospy.loginfo("BROADCASTING")

    def remove_outlier(self, data):
        # remove outlier using one class SVM
        clf = OneClassSVM(gamma='auto')
        clf.fit(data)
        indx = clf.predict(data)
        data = data[indx == 1]
        return data

    def best_fit_transform(self, model, target):
        # calculate centroid
        c_target = np.mean(target, axis=0)
        c_model = np.mean(model, axis=0)

        # de mean
        m_target = (target-c_target).T
        m_model = (model-c_model).T

        # calculate correlation matrix
        H = np.matmul(m_target, m_model.T)

        # SVD decompose
        U, D, V = np.linalg.svd(H)

        # get Rotation & Translation
        rotation = np.dot(U, V)
        translation = c_model - np.dot(rotation, c_target)

        # combine matrix
        transform = np.identity(target.shape[1]+1)
        transform[0:target.shape[1], 0:target.shape[1]] = rotation
        transform[0:target.shape[1], target.shape[1]] = translation

        return transform

    def cb_detection(self, msg):
        # check there is more than 3 detections
        # sort the detections by id
        # check detection id 4 5 6 exist
        if len(msg.detections) >= 3:
            detections = sorted(msg.detections, key=lambda s: s.id[0])
            id_check = 4
            for i in range(3):
                if i != detections[i].id[0]-4:
                    print('missing gate detection')
                    return
                else:
                    id_check += 1

            # calculate tags position relative to car
            tag_poses = []
            for i in range(3):
                p = detections[i].pose.pose.pose.position
                t = detections[i].pose.pose.pose.orientation
                pos = [p.x, p.y, p.z]
                tran = [t.x, t.y, t.z, t.w]

                tag_tf = tf.transformations.concatenate_matrices(
                    tf.transformations.translation_matrix(pos), tf.transformations.quaternion_matrix(tran))
                tag_position = np.dot(self.camera2foot, tag_tf)[0:3, 3]
                tag_poses.append(tag_position)
            tag_poses = np.array(tag_poses)

            # ICP
            total_tf = np.identity(4)
            for i in range(self.iteration):
                # find best transform in this iteration
                transform = self.best_fit_transform(self.gate_poses, tag_poses)
                # accumulate total transform
                total_tf = np.dot(transform, total_tf)
                # transform tag_poses
                for i in range(tag_poses.shape[0]):
                    tag_poses[i] = transform[0:3, 3] + \
                        np.dot(transform[0:3, 0:3], tag_poses[i])

            # transform matrix to quarternion
            quat = tf.transformations.quaternion_from_matrix(total_tf)
            quat = quat/np.linalg.norm(quat)  # normalization
            translation = total_tf[0:3, 3]

            # save tf
            if translation[0] < 0:
                self.translations[self.count] = np.array(translation)
                self.orientations[self.count] = np.array(quat)
                self.count += 1
                self.broadcaster.sendTransform(
                    translation, quat, rospy.Time.now(), "map", "tmp_global")

            
            if self.count == MAX_DETECT:
                # remove outlier using one class SVM
                self.translations = self.remove_outlier(self.translations)
                self.orientations = self.remove_outlier(self.orientations)

                # calculate mean
                self.trans_mean = np.mean(self.translations, axis=0)
                self.orien_mean = np.mean(self.orientations, axis=0)
                self.orien_mean = self.orien_mean/np.linalg.norm(self.orien_mean)  # normalization

                rospy.loginfo('%d translation is used' %
                              self.translations.shape[0])
                rospy.loginfo('%d orientations is used' %
                              self.orientations.shape[0])
                # stop subscriber
                self.sub_detection.unregister()

                # start timer to publish tf
                self.tiemr = rospy.Timer(rospy.Duration(0.02), self.pub_tf)


if __name__ == "__main__":
    rospy.init_node('icp_gate')
    icp = ICP()
    rospy.spin()
