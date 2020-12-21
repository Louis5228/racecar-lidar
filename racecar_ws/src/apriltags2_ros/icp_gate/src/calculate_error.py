import rospy
import tf
import numpy as np

tag7 = [4.024, 20.837, 0.9975]
tag27 = [15.8315, 0.8065, 0.991]

test_num = 5000
rospy.init_node('cal_error')
listener = tf.TransformListener()
total = np.zeros((test_num,3))
for i in range(test_num):
    try:
        listener.waitForTransform('global', 'tag7',rospy.Time(0),rospy.Duration(1))
        (trans_c, rot_c) = listener.lookupTransform('global', 'tag7', rospy.Time(0))
        print(trans_c)
        total[i] = trans_c
    except:
        print("missing")
        pass

final = np.mean(total,axis=0)

print(final)

print(final - tag7)
