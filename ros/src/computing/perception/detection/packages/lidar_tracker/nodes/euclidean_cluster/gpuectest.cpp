#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <vector>
#include <iostream>
#include <time.h>
#include <sys/time.h>

#include "gpuectest.h"
#include "euclidean_cluster/include/euclidean_cluster.h"

#define SAMPLE_DIST_ (1024.0)
#define SAMPLE_RAND_ (1024)
#define SAMPLE_SIZE_ (32768)
#define SAMPLE_SIZE_F_ (32768.0)

void GPUECTest::sparseGraphTest()
{
	sparseGraphTest100();

	sparseGraphTest875();

	sparseGraphTest75();

	sparseGraphTest625();

	sparseGraphTest50();

	sparseGraphTest375();

	sparseGraphTest25();

	sparseGraphTest125();

	sparseGraphTest0();
}

void GPUECTest::sparseGraphTest100()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 100%
	for (int i = 0; i < sample_cloud->points.size(); i++) {
		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.y = sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.z = sample_dist / SAMPLE_DIST_;

		sample_cloud->points[i] = sample_point;
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 100% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 100% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 100% - Edge-based: " << timeDiff(start, end) << std::endl;
}

void GPUECTest::sparseGraphTest875()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	int part_size;
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 87.5%

	part_size = (SAMPLE_SIZE_F_ + sqrt(SAMPLE_SIZE_F_ * SAMPLE_SIZE_F_ - 2 * 0.125 * SAMPLE_SIZE_F_ * (SAMPLE_SIZE_F_ - 1))) / 2;

	for (int i = 0; i < part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.y = sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.z = sample_dist / SAMPLE_DIST_;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	for (int i = 0; i < SAMPLE_SIZE_ - part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.y = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.z = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 87.5% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 87.5% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 87.5% - Edge-based: " << timeDiff(start, end) << std::endl;
}

void GPUECTest::sparseGraphTest75()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	int part_size;
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 75%
	part_size = (SAMPLE_SIZE_F_ + sqrt(SAMPLE_SIZE_F_ * SAMPLE_SIZE_F_ - 2 * 0.25 * SAMPLE_SIZE_F_ * (SAMPLE_SIZE_F_ - 1))) / 2;

	for (int i = 0; i < part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.y = sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.z = sample_dist / SAMPLE_DIST_;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	for (int i = 0; i < SAMPLE_SIZE_ - part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.y = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.z = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 75% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 75% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 75% - Edge-based: " << timeDiff(start, end) << std::endl;

}

void GPUECTest::sparseGraphTest625()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	int part_size;
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 62.5%
	part_size = (SAMPLE_SIZE_F_ + sqrt(SAMPLE_SIZE_F_ * SAMPLE_SIZE_F_ - 2 * 0.375 * SAMPLE_SIZE_F_ * (SAMPLE_SIZE_F_ - 1))) / 2;

	for (int i = 0; i < part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.y = sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.z = sample_dist / SAMPLE_DIST_;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	for (int i = 0; i < SAMPLE_SIZE_ - part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.y = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.z = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 62.5% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 62.5% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 62.5% - Edge-based: " << timeDiff(start, end) << std::endl;
}

void GPUECTest::sparseGraphTest50()
{
	// Density 50%
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	int part_size;
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	part_size = (SAMPLE_SIZE_F_ + sqrt(SAMPLE_SIZE_F_ * SAMPLE_SIZE_F_ - 2 * 0.5 * SAMPLE_SIZE_F_ * (SAMPLE_SIZE_F_ - 1))) / 2;

	for (int i = 0; i < part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.y = sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.z = sample_dist / SAMPLE_DIST_;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	for (int i = 0; i < SAMPLE_SIZE_ - part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.y = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.z = d_th * 10 + sample_dist / SAMPLE_DIST_;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 50% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 50% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 50% - Edge-based: " << timeDiff(start, end) << std::endl;

}

void GPUECTest::sparseGraphTest375()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	int part_size;
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 37.5%

	part_size = SAMPLE_SIZE_ / 2;

	for (int i = 0; i < part_size; i++) {
		int pid = 0;

		while (status[pid]) {
			pid = rand() % SAMPLE_SIZE_;
		}

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.x = sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.y = sample_dist / SAMPLE_DIST_;

		sample_dist = rand() % SAMPLE_RAND_;
		sample_point.z = sample_dist / SAMPLE_DIST_;

		sample_cloud->points[pid] = sample_point;
		status[pid] = true;
	}

	part_size = SAMPLE_SIZE_ / 4;

	for (int i = 1; i <= 2; i++) {
		for (int j = 0; j < part_size; j++) {
			int pid = 0;

			while (status[pid]) {
				pid = rand() % SAMPLE_SIZE_;
			}

			sample_dist = rand() % SAMPLE_RAND_;
			sample_point.x = d_th * i * 10 + sample_dist / SAMPLE_DIST_;


			sample_dist = rand() % SAMPLE_RAND_;
			sample_point.y = d_th * i * 10 + sample_dist / SAMPLE_DIST_;

			sample_dist = rand() % SAMPLE_RAND_;
			sample_point.z = d_th * i * 10 + sample_dist / SAMPLE_DIST_;

			sample_cloud->points[pid] = sample_point;
			status[pid] = true;
		}
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 37.5% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 37.5% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 37.5% - Edge-based: " << timeDiff(start, end) << std::endl;

}

void GPUECTest::sparseGraphTest25()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	int part_size;
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 25%
	part_size = SAMPLE_SIZE_ / 4;

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < part_size; j++) {
			int pid = 0;

			while (status[pid]) {
				pid = rand() % SAMPLE_SIZE_;
			}

			sample_dist = rand() % SAMPLE_RAND_;
			sample_point.x = d_th * i * 10 + sample_dist / SAMPLE_DIST_;

			sample_dist = rand() % SAMPLE_RAND_;
			sample_point.y = d_th * i * 10 + sample_dist / SAMPLE_DIST_;

			sample_dist = rand() % SAMPLE_RAND_;
			sample_point.z = d_th * i * 10 + sample_dist / SAMPLE_DIST_;

			sample_cloud->points[pid] = sample_point;
			status[pid] = true;
		}
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 25% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 25% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 25% - Edge-based: " << timeDiff(start, end) << std::endl;

}

void GPUECTest::sparseGraphTest125()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;
	float sample_dist;
	std::vector<bool> status(SAMPLE_SIZE_, false);
	int part_size;
	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 12.5%
	part_size = SAMPLE_SIZE_ / 8;

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < part_size; j++) {
			int pid = 0;

			while (status[pid]) {
				pid = rand() % SAMPLE_SIZE_;
			}

			sample_dist = rand() % SAMPLE_RAND_;
			sample_point.x = d_th * i * 10 + sample_dist / SAMPLE_DIST_;

			sample_dist = rand() % SAMPLE_RAND_;
			sample_point.y = d_th * i * 10 + sample_dist / SAMPLE_DIST_;

			sample_dist = rand() % SAMPLE_RAND_;
			sample_point.z = d_th * i * 10 + sample_dist / SAMPLE_DIST_;

			sample_cloud->points[pid] = sample_point;
			status[pid] = true;
		}
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 12.5% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 12.5% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 12.5% - Edge-based: " << timeDiff(start, end) << std::endl;
}

void GPUECTest::sparseGraphTest0()
{
	GpuEuclideanCluster2 test_sample;

	srand(time(NULL));

	// Fix at SAMPLE_SIZE_ points, density varies from 100% to 0%
	pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>());

	sample_cloud->points.insert(sample_cloud->points.begin(), SAMPLE_SIZE_, pcl::PointXYZ(0, 0, 0));

	pcl::PointXYZ sample_point(0, 0, 0);
	float d_th = 1.0;

	struct timeval start, end;


	// Set clustering parameters
	test_sample.setBlockSizeX(1024);
	test_sample.setThreshold(d_th);

	// Density 0%
	sample_point.x = sample_point.y = sample_point.z = 0;

	for (int i = 0; i < sample_cloud->points.size(); i++) {
		if (i % 3 == 0)
			sample_point.x += d_th;
		else if (i % 3 == 1)
			sample_point.x += d_th;
		else
			sample_point.z += d_th;

		sample_cloud->points[i] = sample_point;
	}

	test_sample.setInputPoints(sample_cloud);

	gettimeofday(&start, NULL);
	test_sample.extractClusters();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 0% - Matrix-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters2();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 0% - Vertex-based: " << timeDiff(start, end) << std::endl;

	gettimeofday(&start, NULL);
	test_sample.extractClusters3();
	test_sample.getOutput();
	gettimeofday(&end, NULL);

	std::cout << "Density 0% - Edge-based: " << timeDiff(start, end) << std::endl;
}

