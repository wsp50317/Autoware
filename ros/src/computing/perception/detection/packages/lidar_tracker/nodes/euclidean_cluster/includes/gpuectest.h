#ifndef GPU_EC_TEST_
#define GPU_EC_TEST_

#include <iostream>

class GPUECTest {
public:
	GPUECTest();

	// Sparse graph test
	static void sparseGraphTest();

	// Cluster number variation
	static void clusterNumVariationTest();

	// Load-imbalance test
	static void imbalanceTest();

private:

	static void sparseGraphTest100();

	static void sparseGraphTest875();

	static void sparseGraphTest75();

	static void sparseGraphTest625();

	static void sparseGraphTest50();

	static void sparseGraphTest375();

	static void sparseGraphTest25();

	static void sparseGraphTest125();

	static void sparseGraphTest0();

	static void clusterNumVariationTest(int cluster_num);
};

#ifndef timeDiff
#define timeDiff(start, end) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec))
#endif

#endif
