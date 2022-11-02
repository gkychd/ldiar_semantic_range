/*
   Copyright (c) 2020 WX96

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "ground_remove.h"


static int64_t gtm() {
	struct timeval tm;
	gettimeofday(&tm, 0);
	// return ms
	int64_t re = (((int64_t) tm.tv_sec) * 1000 * 1000 + tm.tv_usec);
	return re;
}

bool point_cmp(PointCloudXYZRDP a, PointCloudXYZRDP b) {
	return a.z < b.z;
}

GroundRemove::GroundRemove(int num_iter, int num_lpr, double th_seeds,
		double th_dist) {
	num_iter_ = num_iter;
	num_lpr_ = num_lpr;
	th_seeds_ = th_seeds;
	th_dist_ = th_dist;
}

void GroundRemove::extract_initial_seeds_(
		const pcl::PointCloud<PointCloudXYZRDP>& p_sorted,
		pcl::PointCloud<PointCloudXYZRDP>& g_seeds_pc) {
	// LPR is the mean of low point representative
	double sum = 0;
	int cnt = 0;
	// Calculate the mean height value. 取20个z轴最小的点
	for (int i = 0; i < p_sorted.points.size() && cnt < num_lpr_; ++i) {
		sum += p_sorted.points[i].z;
		cnt++;
	}
	double lpr_height = cnt != 0 ? sum / cnt : 0; // in case divide by 0 取20个点的z坐标的平均值
	std::cout << "average min height: " << lpr_height << std::endl;
	g_seeds_pc.clear();
	// iterate pointcloud, filter those height is less than lpr.height+th_seeds_
	for (int i = 0; i < p_sorted.points.size(); ++i) {
		if (p_sorted.points[i].z < lpr_height + th_seeds_) {
			g_seeds_pc.points.push_back(p_sorted.points[i]);
		}
	}
	// return seeds points
}

void GroundRemove::estimate_plane_(
		const pcl::PointCloud<PointCloudXYZRDP>& g_ground_pc) {
	// Create covarian matrix in single pass.
	// TODO: compare the efficiency.
	Eigen::Matrix3f cov;
	Eigen::Vector4f pc_mean;

	pcl::computeMeanAndCovarianceMatrix(g_ground_pc, cov, pc_mean);

	//computeMeanAndCovarianceMatrix(g_ground_pc, conv, mean);
	// Singular Value Decomposition: SVD
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov, Eigen::DecompositionOptions::ComputeFullU);

	std::cout << "centor point: " << "(" <<  pc_mean(0) << "," << pc_mean(1) << "," << pc_mean(2) << ")" << std::endl;
	// use the least singular vector as normal
	normal_ = Eigen::MatrixXf(3,1);
	normal_.fill(0.0);
	normal_ = (svd.matrixU().col(2));//取最小特征值对应的特征向量为法向量
	// mean ground seeds value
	//Eigen::Vector3f seeds_mean = pc_mean.head<3>();

	// according to normal.T*[x,y,z] = -d
	//float d_ = -(normal_.transpose()*seeds_mean)(0,0);

	float d_ = -(normal_(0, 0) * pc_mean(0) + normal_(1, 0) * pc_mean(1)
			+ normal_(2, 0) * pc_mean(2));
	//d_ = d_ - 1.73; //1.73为传感器高度
	std::cout << "D0: " << d_ << std::endl;
	// set distance threhold to `th_dist - d`
	th_dist_d_ = th_dist_ - d_;

	// return the equation parameters
}

void GroundRemove::RemoveGround_Thread(pcl::PointCloud<PointCloudXYZRDP>& cloudIn,
		pcl::PointCloud<PointCloudXYZRDP>& cloudgc,
		pcl::PointCloud<PointCloudXYZRDP>& cloudngc,
		pcl::PointCloud<PointCloudXYZRDP>& g_ground_pc1,
		pcl::PointCloud<PointCloudXYZRDP>& g_not_ground_pc1) {

	std::lock_guard < std::mutex > lock(regionmutex);
	pcl::PointCloud<PointCloudXYZRDP>::Ptr g_seeds_pc(
			new pcl::PointCloud<PointCloudXYZRDP>());

	std::sort(cloudIn.points.begin(), cloudIn.points.end(), point_cmp);//根据点的z轴坐标从小到大排列
	//根据z最小的20个点的平均值 z_平均，将z < z_平均+1.0的点视为新的种子点，也就其他不满足该要求的点不进行以下流程
	extract_initial_seeds_(cloudIn, *g_seeds_pc);
	cloudgc = *g_seeds_pc;

	for (int i = 0; i < num_iter_; ++i) {
		//将剩余的种子点 进行平面拟合 Ax+By+Cz+D=0
		//将这些点的中点带入平面方程得到D0，然后得到deltaD = 0.15 - D0
		estimate_plane_(cloudgc);
		cloudgc.clear();
		cloudngc.clear();
		//拟合平面的法向量
		float xd = normal_(0, 0);
		float yd = normal_(1, 0);
		float zd = normal_(2, 0);
		std::cout << "normal vector: " << "(" <<  xd << "," << yd << "," << zd << ")" << std::endl;
		for (auto p : cloudIn.points) {
			float distance = p.x * xd + p.y * yd + p.z * zd;
			if (distance < th_dist_d_) {
				//g_all_pc->points[r].label = 1u;// means ground
				cloudgc.points.push_back(p);
			} else {
				//g_all_pc->points[r].label = 0u;// means not ground and non clusterred
				cloudngc.points.push_back(p);
			}

		}

	}
	//copy了一份
	for (int k = 0; k < cloudgc.points.size(); ++k) {

		g_ground_pc1.points.push_back(cloudgc.points[k]);
	}

	for (int k = 0; k < cloudngc.points.size(); ++k) {

		g_not_ground_pc1.points.push_back(cloudngc.points[k]);
	}

}

void GroundRemove::RemoveGround(pcl::PointCloud<PointCloudXYZRDP>& cloudIn,
		pcl::PointCloud<PointCloudXYZRDP>& g_ground_pc,
		pcl::PointCloud<PointCloudXYZRDP>& g_not_ground_pc) {
	pcl::PointCloud<PointCloudXYZRDP>::Ptr g_seeds_region1(
			new pcl::PointCloud<PointCloudXYZRDP>());
	pcl::PointCloud<PointCloudXYZRDP>::Ptr g_seeds_region2(
			new pcl::PointCloud<PointCloudXYZRDP>());
	pcl::PointCloud<PointCloudXYZRDP>::Ptr g_seeds_region3(
			new pcl::PointCloud<PointCloudXYZRDP>());
	pcl::PointCloud<PointCloudXYZRDP>::Ptr g_ground_pc1(
			new pcl::PointCloud<PointCloudXYZRDP>());
	pcl::PointCloud<PointCloudXYZRDP>::Ptr g_not_ground_pc1(
			new pcl::PointCloud<PointCloudXYZRDP>());

	float xmin = -35, xmax = 35, ymin = -30, ymax = 30, zmin = -2.0, zmax = 5.0;
	float regionsize = (ymax - ymin) / 3;
	for (int i = 0; i < cloudIn.points.size(); ++i) {
		if (cloudIn.points[i].z < 0.75) {
			//根据y值，将点云分成三部分
			if (cloudIn.points[i].y < ymax - 0 * regionsize
					&& cloudIn.points[i].y > ymax - 1 * regionsize) {
				g_seeds_region1->points.push_back(cloudIn.points[i]);
			}
			if (cloudIn.points[i].y < ymax - 1 * regionsize
					&& cloudIn.points[i].y > ymax - 2 * regionsize) {
				g_seeds_region2->points.push_back(cloudIn.points[i]);
			}
			if (cloudIn.points[i].y < ymax - 2 * regionsize
					&& cloudIn.points[i].y > ymax - 3 * regionsize) {
				g_seeds_region3->points.push_back(cloudIn.points[i]);
			}
		} else {
			//如果点的y<-30或y>30则直接视为非地面点
			g_not_ground_pc.points.push_back(cloudIn.points[i]);
		}

	}

	cloudIn.clear();

	std::vector<pcl::PointCloud<PointCloudXYZRDP>::Ptr> pcregion(3);

	pcregion[0] = g_seeds_region1;
	pcregion[1] = g_seeds_region2;
	pcregion[2] = g_seeds_region3;

	std::vector<std::thread> thread_vec(num_seg_);

	for (int ri = 0; ri < num_seg_; ++ri) {

		thread_vec[ri] = std::thread(&GroundRemove::RemoveGround_Thread, this,
				std::ref(*pcregion[ri]), std::ref(*g_ground_pc1),
				std::ref(*g_not_ground_pc1), std::ref(g_ground_pc),
				std::ref(g_not_ground_pc));

	}

	for (auto it = thread_vec.begin(); it != thread_vec.end(); ++it) {
		it->join();
	}

}


/*template<typename PointInT, typename PointOutT>
void CloudFilter(const pcl::PointCloud<PointInT>& cloudIn,
		pcl::PointCloud<PointOutT>& cloudOut, float x_min, float x_max,
		float y_min, float y_max, float z_min, float z_max) {
	cloudOut.header = cloudIn.header;
	cloudOut.sensor_orientation_ = cloudIn.sensor_orientation_;
	cloudOut.sensor_origin_ = cloudIn.sensor_origin_;
	cloudOut.points.clear();
	//1) set parameters for removing cloud reflect on ego vehicle
	float x_limit_min = -1.8, x_limit_max = 1.8, y_limit_forward = 5.0,
			y_limit_backward = -4.5;
	//2 apply the filter
	for (int i = 0; i < cloudIn.size(); ++i) {
		float x = cloudIn.points[i].x;
		float y = cloudIn.points[i].y;
		float z = cloudIn.points[i].z;
		// whether on ego vehicle
		if ((x > x_limit_min && x < x_limit_max && y > y_limit_backward
				&& y < y_limit_forward))
			continue;
		if ((x > x_min && x < x_max && y > y_min && y < y_max && z > z_min
				&& z < z_max)) {

			cloudOut.points.push_back(cloudIn.points[i]);

		}

	}
}

void Cloudcolor(const pcl::PointCloud<PointCloudXYZRDPI>& cloudIn,
		const pcl::PointCloud<PointCloudXYZRDPI>& cloudground,
		pcl::PointCloud<PointCloudXYZRDPRGB>& cloudOut) {

	int groundsize = cloudground.points.size();
	int ngroundsize = cloudIn.points.size();
	int size = groundsize + ngroundsize;

	for (int i = 0; i < ngroundsize; ++i) {
		PointCloudXYZRDPRGB p;
		p.x = cloudIn.points[i].x;
		p.y = cloudIn.points[i].y;
		p.z = cloudIn.points[i].z;
		p.r = 225;
		p.g = 0;
		p.b = 0;
		cloudOut.points.push_back(p);
	}

	for (int i = 0; i < groundsize; ++i) {
		PointCloudXYZRDPRGB p;
		p.x = cloudground.points[i].x;
		p.y = cloudground.points[i].y;
		p.z = cloudground.points[i].z;
		p.r = 225;
		p.g = 255;
		p.b = 255;
		cloudOut.points.push_back(p);
	}
}

class SubscribeAndPublish {
 public:
 SubscribeAndPublish(ros::NodeHandle nh, std::string lidar_topic_name,
 std::string imu_topic_name);

 void callback(const sensor_msgs::PointCloud2ConstPtr& cloudmsg,
 const sensor_driver_msgs::GpswithHeadingConstPtr& gps_msg) {
 pcl::PointCloud<PointCloudXYZRDPI>::Ptr cloud(
 new pcl::PointCloud<PointCloudXYZRDPI>);
 pcl::PointCloud<PointCloudXYZRDPI>::Ptr cloud_t(
 new pcl::PointCloud<PointCloudXYZRDPI>);
 pcl::PointCloud<PointCloudXYZRDPI>::Ptr cloud_f(
 new pcl::PointCloud<PointCloudXYZRDPI>);
 pcl::PointCloud<PointCloudXYZRDPRGB>::Ptr cloud_color(
 new pcl::PointCloud<PointCloudXYZRDPRGB>);

 pcl::PointCloud<PointCloudXYZRDPI>::Ptr g_ground_pc(
 new pcl::PointCloud<PointCloudXYZRDPI>());
 pcl::PointCloud<PointCloudXYZRDPI>::Ptr g_not_ground_pc(
 new pcl::PointCloud<PointCloudXYZRDPI>());
 pcl::fromROSMsg(*cloudmsg, *cloud);

 float xmin = -35, xmax = 35, ymin = -30, ymax = 30, zmin = -1.0, zmax =
 3.0;

 CloudFilter(*cloud, *cloud_t, xmin, xmax, ymin, ymax, zmin, zmax);
 TransformKittiCloud( *cloud_t,*cloud_f);

 cloud->clear();
 int64_t tm0 = gtm();
 GroundRemove grobject(3,20,1.0,0.15);
 grobject.RemoveGround(*cloud_f,*g_ground_pc,*g_not_ground_pc);
 int64_t tm1 = gtm();
 printf("[INFO]region build cast time:%ld us\n", tm1 - tm0);

 Cloudcolor(*g_ground_pc, *g_not_ground_pc, *cloud_color);

 cloud_color->height = 1;
 cloud_color->width = cloud_color->points.size();
 cloud_color->is_dense = false;    //最终优化结果

 sensor_msgs::PointCloud2 ros_cloud;
 pcl::toROSMsg(*cloud_color, ros_cloud);
 ros_cloud.header.frame_id = "global_init_frame";
 pub_.publish(ros_cloud);
 //pcl::io::savePCDFileASCII<PointCloudXYZRDPRGB> ("test_simple.pcd", *cloud_color);
 //pcl::visualization::CloudViewer viewer("PCD2");
 //viewer.showCloud(cloud_color);

 }
 private:
 ros::NodeHandle n_;
 ros::Publisher pub_;
 ros::Subscriber sub_;
 message_filters::Subscriber<sensor_msgs::PointCloud2> Sub_Lidar;
 //message_filters::Subscriber<sensor_driver_msgs::GpswithHeading> Sub_IMU;
 message_filters::Subscriber<sensor_driver_msgs::GpswithHeading> Sub_IMU;
 typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
 sensor_driver_msgs::GpswithHeading> MySyncPolicy;
 // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
 Synchronizer<MySyncPolicy> sync;

 };

 SubscribeAndPublish::SubscribeAndPublish(ros::NodeHandle nh,
 std::string lidar_topic_name, std::string imu_topic_name) :
 n_(nh), Sub_Lidar(nh, lidar_topic_name, 10), Sub_IMU(nh, imu_topic_name,
 20), sync(MySyncPolicy(10), Sub_Lidar, Sub_IMU) {
 //Topic you want to publish
 pub_ = nh.advertise < sensor_msgs::PointCloud2 > ("/groundremove", 1);

 //Topic you want to subscribe
 //sub_ = n_.subscribe("lidar_cloud_calibrated", 1, &SubscribeAndPublish::callback, this);
 // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
 sync.registerCallback(
 boost::bind(&SubscribeAndPublish::callback, this, _1, _2));
 }


 int main(int argc, char** argv) {

 ros::init(argc, argv, "ground_node");
 SubscribeAndPublish SAPObject(ros::NodeHandle(), "lidar_cloud_calibrated",
 "gpsdata");
 ROS_INFO("waiting for data!");
 ros::spin();

 return 0;
 }*/

