#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/point_cloud_conversion.h>

ros::Publisher pub;

void cloud_cb (const sensor_msgs::PointCloudConstPtr& input)
{
	 ROS_INFO("function call \n");
  // Create a container for the data.
  sensor_msgs::PointCloud2 output;

  sensor_msgs::PointCloud convert_input = *input;

  // convert PointCloud format
  bool trans_res =  sensor_msgs::convertPointCloudToPointCloud2	(convert_input, output)	;

  // Publish the data.
  pub.publish (output);
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "PointCloud_translator");
  ros::NodeHandle nh;

 	ROS_INFO("receive data \n");
  ros::Subscriber sub = nh.subscribe ("prius/center_laser/scan", 1, cloud_cb);
  pub = nh.advertise<sensor_msgs::PointCloud2> ("points_raw", 1);

  // Spin
  ros::spin ();
}
