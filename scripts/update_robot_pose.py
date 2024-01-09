#!/usr/bin/env python
import rospy
import tf
import numpy as np 

from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose
from nav_msgs.msg import Odometry


class posePublisher:
    def __init__(self):
        self.tag_locations = rospy.get_param("update_robot_pose/tag_locations")    # get tag locations from yaml 
        self.max_detection_dist = rospy.get_param("update_robot_pose/max_detection_dist")
        self.max_linear_vel_x = rospy.get_param("update_robot_pose/max_linear_vel_x")
        self.max_angular_vel_z = rospy.get_param("update_robot_pose/max_angular_vel_z")
        self.xy_tolerance = rospy.get_param("update_robot_pose/xy_tolerance")
        self.yaw_tolerance = rospy.get_param("update_robot_pose/yaw_tolerance")

        self.curr_linear_vel_x = 0
        self.curr_angular_vel_z = 0
        
        self.tag_sub = rospy.Subscriber("tag_detections", AprilTagDetectionArray, self.poseCallback, queue_size=1)
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odomCallback, queue_size=1)
        self.map_base_link_pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=1)

        self.tf_ros = tf.TransformerROS()
        self.tf_listener = tf.TransformListener()

        # get transformation between base_link and usb_cam_link from tf_tree
        self.tf_listener.waitForTransform("base_link", "usb_cam_link", rospy.Time(0), rospy.Duration(1.0))
        (self.base_link_usb_cam_link_t, self.base_link_usb_cam_link_q) = self.tf_listener.lookupTransform("base_link", "usb_cam_link", rospy.Time(0))


    def odomCallback(self, data):
        self.curr_linear_vel_x = abs(data.twist.twist.linear.x)
        self.curr_angular_vel_z = abs(data.twist.twist.angular.z)


    def poseCallback(self, data): 
        # linear and angular velocity threshold for updating the robot's pose
        if data.detections and self.curr_linear_vel_x <= self.max_linear_vel_x and self.curr_angular_vel_z <= self.max_angular_vel_z:
            map_base_link_data = PoseWithCovarianceStamped()
            # map_base_link_data.header.seq = data.header.seq
            map_base_link_data.header.stamp = rospy.Time.now()
            map_base_link_data.header.frame_id = "map"

            # tag w.r.t. map
            tag_detected = []
            for i in range(len(data.detections)):
                # distance between tag and base_link
                if data.detections[i].pose.pose.pose.position.z <= self.max_detection_dist: 
                    tag_detected.append(data.detections[i])
        
            self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
            (map_base_link_actual_t, map_base_link_actual_q) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
            xy_actual = np.array([abs(map_base_link_actual_t[0]), abs(map_base_link_actual_t[1])])            # actual xy position
            (_, _, yaw_actual) = tf.transformations.euler_from_quaternion(map_base_link_actual_q)   # actual yaw angle
    
            if tag_detected:
                tag_sorted = sorted(tag_detected, key=lambda tag_detected: tag_detected.pose.pose.pose.position.z)

                for tag in self.tag_locations:
                    if tag["id"] == int(''.join(map(str, tag_sorted[0].id))): 
                        break
                
                map_tag_t = [tag["x"], tag["y"], tag["z"]]
                map_tag_q = [tag["qx"], tag["qy"], tag["qz"], tag["qw"]]
                map_tag_g = self.tf_ros.fromTranslationRotation(map_tag_t, map_tag_q) 

                # tag w.r.t. usb_cam_link
                usb_cam_link_tag_t = [tag_sorted[0].pose.pose.pose.position.x, tag_sorted[0].pose.pose.pose.position.y, tag_sorted[0].pose.pose.pose.position.z]
                usb_cam_link_tag_q = [tag_sorted[0].pose.pose.pose.orientation.x, tag_sorted[0].pose.pose.pose.orientation.y, tag_sorted[0].pose.pose.pose.orientation.z, tag_sorted[0].pose.pose.pose.orientation.w]
                usb_cam_link_tag_g = self.tf_ros.fromTranslationRotation(usb_cam_link_tag_t, usb_cam_link_tag_q)

                # usb_cam w.r.t. base_link
                base_link_usb_cam_link_g = self.tf_ros.fromTranslationRotation(self.base_link_usb_cam_link_t, self.base_link_usb_cam_link_q)

                # calculation of base_link w.r.t. map
                tag_usb_cam_link_g = np.linalg.inv(usb_cam_link_tag_g)                  # usb_cam_link w.r.t. tag
                usb_cam_link_base_link_g = np.linalg.inv(base_link_usb_cam_link_g)      # base_link w.r.t. usb_cam_link

                map_base_link_g = np.matmul(map_tag_g, np.matmul(tag_usb_cam_link_g, usb_cam_link_base_link_g))

                # get rotation and translation matrix from transformation matrix 
                map_base_link_t = map_base_link_g[:3, 3]
                # map_base_link_R = map_base_link_g[:3, :3]
                map_base_link_q = tf.transformations.quaternion_from_matrix(map_base_link_g)

                xy_detect = np.array([abs(map_base_link_t[0]), abs(map_base_link_t[1])])
                (_, _, yaw_detect) = tf.transformations.euler_from_quaternion(map_base_link_q)

                xy_diff = np.linalg.norm(xy_actual - xy_detect)
                yaw_diff = abs(abs(yaw_actual) - abs(yaw_detect))

                # xy and yaw tolerance for updating the robot's pose
                if xy_diff > self.xy_tolerance or yaw_diff > self.yaw_tolerance:
                    map_base_link_data.pose.pose.position.x = map_base_link_t[0]
                    map_base_link_data.pose.pose.position.y = map_base_link_t[1]
                    map_base_link_data.pose.pose.position.z = map_base_link_t[2]
                    map_base_link_data.pose.pose.orientation.x = map_base_link_q[0]
                    map_base_link_data.pose.pose.orientation.y = map_base_link_q[1]
                    map_base_link_data.pose.pose.orientation.z = map_base_link_q[2]
                    map_base_link_data.pose.pose.orientation.w = map_base_link_q[3]

                    self.map_base_link_pub.publish(map_base_link_data)
    

if __name__ == '__main__':
    try:
        rospy.init_node('posePublisher', anonymous=True)
        posePublisher = posePublisher()
        rospy.loginfo("Getting transformation from base_link to map.")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
