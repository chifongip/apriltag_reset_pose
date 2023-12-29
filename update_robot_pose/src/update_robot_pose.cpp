#include "ros/ros.h"
#include "tf/tf.h"
#include "tf/transform_listener.h"
#include "apriltag_ros/AprilTagDetection.h"
#include "apriltag_ros/AprilTagDetectionArray.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "nav_msgs/Odometry.h"

#include <XmlRpcException.h>

#include <iostream>
#include <map>
#include <vector>
#include <cmath>

using namespace std;

class resetPose
{
public:
    resetPose()
    {
        nh.getParam("update_robot_pose/tag_locations", tag_locations);
        nh.getParam("update_robot_pose/max_detection_dist", max_detection_dist);
        nh.getParam("update_robot_pose/max_linear_vel_x", max_linear_vel_x);
        nh.getParam("update_robot_pose/max_angular_vel_z", max_angular_vel_z);
        nh.getParam("update_robot_pose/xy_tolerance", xy_tolerance);
        nh.getParam("update_robot_pose/yaw_tolerance", yaw_tolerance);

        // get transformation matrix of usb_cam_link w.r.t base_link from tf_tree
        tf_listener.waitForTransform("base_link", "usb_cam_link", ros::Time(0), ros::Duration(1.0));
        tf_listener.lookupTransform("base_link", "usb_cam_link", ros::Time(0), base_link_usb_cam_link_g);

        tag_detections_sub = nh.subscribe("tag_detections", 10, &resetPose::poseCallback, this);
        odom_sub = nh.subscribe("odom", 10, &resetPose::odomCallback, this);
        initialpose_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("initialpose", 10);

        cout << "success initialization." << endl;

        // for(int i = 0; i < tag_locations.size(); i++)
        // {
        //     for(const auto& elem: tag_locations[i])
        //     {
        //         cout << elem.first << ": " << elem.second << endl;
        //     }
        // }
    }

    static bool compare_dist_from_tag(const apriltag_ros::AprilTagDetection& a, const apriltag_ros::AprilTagDetection& b);

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        curr_linear_vel_x = msg->twist.twist.linear.x;
        curr_angular_vel_z = msg->twist.twist.angular.z;
    }

    void poseCallback(const apriltag_ros::AprilTagDetectionArray::ConstPtr& msg)
    {
        // linear and angular velocity threshold for updating the robot's pose
        if(!msg->detections.empty() && curr_linear_vel_x <= max_linear_vel_x && curr_angular_vel_z <= max_angular_vel_z)
        {
            geometry_msgs::PoseWithCovarianceStamped map_base_link_data;
            map_base_link_data.header.seq = msg->header.seq;
            map_base_link_data.header.stamp = msg->header.stamp;
            map_base_link_data.header.frame_id = "map";

            std::vector<apriltag_ros::AprilTagDetection> tag_detected;
            for(int i = 0; i < msg->detections.size(); i++)
            {
                // distance between tag and base_link
                if(msg->detections[i].pose.pose.pose.position.z <= max_detection_dist)
                {
                    tag_detected.push_back(msg->detections[i]);
                }
            }
            
            tf_listener.waitForTransform("map", "base_link", ros::Time(0), ros::Duration(1.0));
            tf_listener.lookupTransform("map", "base_link", ros::Time(0), map_base_link_actual_g);
            
            xy_actual = {map_base_link_actual_g.getOrigin().x(), map_base_link_actual_g.getOrigin().y()};   // actual xy position
            // map_base_link_actual_q = map_base_link_actual_g.getRotation();
            tf::Matrix3x3(map_base_link_actual_g.getRotation()).getRPY(roll_actual, pitch_actual, yaw_actual);                              // actual yaw angle
            
            if(!tag_detected.empty())
            {
                std::sort(tag_detected.begin(), tag_detected.end(), compare_dist_from_tag);

                usb_cam_link_tag_g.setOrigin(tf::Vector3(tag_detected[0].pose.pose.pose.position.x, tag_detected[0].pose.pose.pose.position.y, tag_detected[0].pose.pose.pose.position.z));
                usb_cam_link_tag_g.setRotation(tf::Quaternion(tag_detected[0].pose.pose.pose.orientation.x, tag_detected[0].pose.pose.pose.orientation.y, tag_detected[0].pose.pose.pose.orientation.z, tag_detected[0].pose.pose.pose.orientation.w));

                int id = 0;
                for(auto d: tag_detected[0].id)
                {
                    id = id * 10 + d;
                }

                for(int i = 0; i < tag_locations.size(); i++)
                {
                    if((int) tag_locations[i]["id"] == id)
                    {
                        map_tag_g.setOrigin(tf::Vector3(tag_locations[i]["x"], tag_locations[i]["y"], tag_locations[i]["z"]));
                        map_tag_g.setRotation(tf::Quaternion(tag_locations[i]["qx"], tag_locations[i]["qy"], tag_locations[i]["qz"], tag_locations[i]["qw"]));
                    }
                }

                // calculation of base_link w.r.t. map
                tag_usb_cam_link_g = usb_cam_link_tag_g.inverse();                  // usb_cam_link w.r.t. tag
                usb_cam_link_base_link_g = base_link_usb_cam_link_g.inverse();      // base_link w.r.t. usb_cam_link
                
                map_base_link_g = map_tag_g * tag_usb_cam_link_g * usb_cam_link_base_link_g;

                // get rotation and translation matrix from transformation matrix 
                xy_detect = {map_base_link_g.getOrigin().x(), map_base_link_g.getOrigin().y()};
                // map_base_link_q = map_base_link_g.getRotation();
                tf::Matrix3x3(map_base_link_g.getRotation()).getRPY(roll_detect, pitch_detect, yaw_detect);

                xy_diff = sqrt(pow((xy_actual[0] - xy_detect[0]), 2) +  pow((xy_actual[1] - xy_detect[1]), 2)); 
                yaw_diff = abs(yaw_actual - yaw_detect);

                // xy and yaw tolerance for updating the robot's pose
                if(xy_diff > xy_tolerance || yaw_diff > yaw_tolerance)
                {
                    map_base_link_data.pose.pose.position.x = map_base_link_g.getOrigin().x();
                    map_base_link_data.pose.pose.position.y = map_base_link_g.getOrigin().y();
                    map_base_link_data.pose.pose.position.z = map_base_link_g.getOrigin().z();
                    map_base_link_data.pose.pose.orientation.x = map_base_link_g.getRotation().x();
                    map_base_link_data.pose.pose.orientation.y = map_base_link_g.getRotation().y();
                    map_base_link_data.pose.pose.orientation.z = map_base_link_g.getRotation().z();
                    map_base_link_data.pose.pose.orientation.w = map_base_link_g.getRotation().w();
                    initialpose_pub.publish(map_base_link_data);
                }
            }
        }
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber tag_detections_sub;
    ros::Subscriber odom_sub;
    ros::Publisher initialpose_pub;

    double curr_linear_vel_x, curr_angular_vel_z;
    double max_detection_dist, max_linear_vel_x, max_angular_vel_z, xy_tolerance, yaw_tolerance;
    XmlRpc::XmlRpcValue tag_locations;

    tf::Transformer tf_tool;
    tf::TransformListener tf_listener;
    tf::StampedTransform base_link_usb_cam_link_g;
    tf::StampedTransform map_base_link_actual_g;
    // tf::Quaternion map_base_link_actual_q;
    // tf::Quaternion map_base_link_q;

    std::vector<double> xy_actual, xy_detect; 
    double roll_actual, pitch_actual, yaw_actual, roll_detect, pitch_detect, yaw_detect;
    double xy_diff, yaw_diff;

    tf::Transform usb_cam_link_tag_g;
    tf::Transform map_tag_g;

    tf::Transform tag_usb_cam_link_g;
    tf::Transform usb_cam_link_base_link_g;

};

bool resetPose::compare_dist_from_tag(const apriltag_ros::AprilTagDetection& a, const apriltag_ros::AprilTagDetection& b)
{
    return a.pose.pose.pose.position.z < b.pose.pose.pose.position.z;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "posePublisher");
    resetPose resetPoseNode;
    ROS_INFO("reset pose node running.");
    ros::spin();

    return 0;
}