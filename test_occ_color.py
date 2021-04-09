from nav_msgs.msg import OccupancyGrid
import rospy
import numpy as np

if __name__ == '__main__':
    rospy.init_node('occ_color_test', anonymous=True)
    pub = rospy.Publisher('testmap', OccupancyGrid, queue_size = 1)
    rate = rospy.Rate(10)
    test_map = OccupancyGrid();
    test_map.header.stamp = rospy.Time.now()
    test_map.header.frame_id = "map"      
    test_map.info.resolution = 1
    test_map.info.width = 16
    test_map.info.height = 16
    test_map.info.origin.position.x = 0
    test_map.info.origin.position.y = 0
    test_map.info.origin.position.z = 0
    test_map.info.origin.orientation.x = 0
    test_map.info.origin.orientation.y = 0
    test_map.info.origin.orientation.z = 0
    test_map.info.origin.orientation.w = 1
    test_map.data = np.full(16*16, -1)
    for i in range(len(test_map.data)):
        test_map.data[i] = i-128

    while not rospy.is_shutdown():
        pub.publish(test_map)
        rate.sleep()