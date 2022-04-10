#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/console/time.h>

#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>

    
pcl::PointCloud<pcl::Normal>::Ptr orgnized_normal_estimation (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,bool test,std::string name)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);

    pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(10.0f);
    ne.setInputCloud(cloud);

	if (test)
	{
		pcl::console::TicToc tt;
		tt.tic();
		ne.compute(*normals);
		std::cout << "orgnized normal estimation cost: " << tt.toc() << "ms" << std::endl << std::endl;
		// save normals to pcd
		pcl::io::savePCDFile("res/"+name+".pcd", *normals);
		std::cout << "save normals to "<<name<<".pcd finish" << std::endl << std::endl;
	}
	else
	{
		ne.compute(*normals);
	}
	
    return normals;
}

pcl::PointCloud<pcl::Normal>::Ptr fast_normal_estimation( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, bool test,std::string name)
{
	// Create a search tree, use KDTreee for non-organized data.
	pcl::search::Search<pcl::PointXYZ>::Ptr tree;

	if (cloud->isOrganized())
	{
		tree.reset(new pcl::search::OrganizedNeighbor<pcl::PointXYZ>());
	}
	else
	{
		tree.reset(new pcl::search::KdTree<pcl::PointXYZ>(false));
	}

	// estimate normals
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

	// Compute normals using both small and large scales at each point
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);
	ne.setSearchMethod(tree);

	/**
	 * NOTE: setting viewpoint is very important, so that we can ensure
	 * normals are all pointed in the same direction!
	 */
	ne.setViewPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

	ne.setRadiusSearch(0.03);
	
	if (test)
	{
		pcl::console::TicToc tt;
		tt.tic();
		ne.compute(*normals);
		std::cout << "fast normal estimation cost: " << tt.toc() << "ms" << std::endl;
		//save normals to pcd 
		pcl::io::savePCDFile(name+".pcd", *normals);
		cout << "save normals to "<<name<<".pcd finish" << endl<<endl;
	}
	else
	{
		ne.compute(*normals);
	}

	return normals;
}

/* @brief long long time do not use */
pcl::PointCloud<pcl::Normal>::Ptr usual_normal_estimation()
{
	// load point cloud
	std::cout << "loading point cloud" << std::endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile("check.pcd", *cloud);
	std::cout << "loading point cloud finish" << std::endl;

	// Create the normal estimation class, and pass the input dataset to it
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);
	
	// Create an empty kdtree representation, and pass it to the normal estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	
	// Output datasets
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

	// Use all neighbors in a sphere of radius 3cm
	ne.setRadiusSearch(0.03);

	

	pcl::console::TicToc tt;
	tt.tic();
	std::cout << "Computing the features" << std::endl;
	// Compute the features
	ne.compute(*cloud_normals);

	std::cout << "usual normal estimation cost: " << tt.toc() << "ms" << std::endl;

	pcl::io::savePCDFile("usual_normals.pcd", *cloud_normals);
	cout << "save normals finish" << endl;

	return cloud_normals;


}