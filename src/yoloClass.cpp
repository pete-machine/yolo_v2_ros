#include <stdio.h>
#include <cv.h>
#include <vector>
#include <highgui.h>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <std_msgs/MultiArrayLayout.h>
//#include <htf_safe_msgs/SAFEObstacleMsg.h>
#include <sensor_msgs/image_encodings.h>
//#include <sensor_msgs/CompressedImage.h>
#include <image_transport/image_transport.h>
#include <boost/algorithm/string.hpp>
//#include <camera_info_manager/camera_info_manager.h>
#include <math.h>


extern "C" {


#include "yoloInterface.h"


#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "list.h"
#include "option_list.h"
#include "blas.h"
}

//using namespace std;
using namespace cv;
float remapYolo2NewObjectTypes[] = {0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
class MyNode {
public:
	MyNode() :
		nh("~"), it(nh) {
			nh.param<std::string>("basedir",basedir,"/folder/of/ros/node");
			nh.param<std::string>("model_cfg",model_cfg,"/cfg/yolo.cfg");
			nh.param<std::string>("weightfile",weightfile,"/weights/yolo.weights");
			nh.param<std::string>("datafile",datafile,"/cfg/coco.data");
			nh.param<bool>("visualize_detections",visualizeDetections,true);

			nh.param<std::string>("topic_name",topic_name,"/usb_cam/image_raw");
			nh.param<float>("threshold",threshold,0.2);
			std::vector<std::string> strParts;
			boost::split(strParts,topic_name,boost::is_any_of("/"));

			model_cfg = basedir+model_cfg;
			weightfile = basedir+weightfile;
			datafile = basedir+datafile;

			// Distance estimate.
			nh.param<double>("FOV_verticalDeg",FOV_verticalDeg,47.0);
			nh.param<double>("FOV_horizontalDeg",FOV_horizontalDeg,83.0);
			nh.param<double>("angleTiltDegrees",angleTiltDegrees,7.0);
			nh.param<double>("cameraHeight",cameraHeight,1.9);

			FOV_verticalRad = FOV_verticalDeg*M_PI/180;
			FOV_horizontalRad = FOV_horizontalDeg*M_PI/180;
			angleTiltRad = angleTiltDegrees*M_PI/180;

			//cinfor_ = boost::shared_ptr<camera_info_manager::CameraInfoManager>(new camera_info_manager::CameraInfoManager(nh, "test", ""));
			//sub_image = it.subscribeCamera(topic_name.c_str(), 1, &MyNode::onImage, this);

			sub_image = it.subscribe(topic_name.c_str(), 1, &MyNode::onImage, this);
			pub_image = it.advertise("imageYolo", 1);
			
			std::vector<std::string> outputTopicTmp;
			outputTopicTmp.push_back("BBox");
			outputTopicTmp.push_back(strParts[1]);
			pub_bb = nh.advertise<std_msgs::Float64MultiArray>(boost::algorithm::join(outputTopicTmp,"/"), 1);

			readyToPublish = 1;

			useRemapping = 1;
			options = (list *)read_data_cfg((char*)datafile.c_str());
			std::string name_list = option_find_str(options, "names", "data/names.list");

			name_list = basedir+ '/' + name_list;
			names = get_labels((char*)name_list.c_str());

			maxDetections = load_yolo_model((char*)model_cfg.c_str(), (char*)weightfile.c_str());
			nClasses = get_nclasses();

			boxes = (box*)calloc(maxDetections, sizeof(box));
			probs = (float**)calloc(maxDetections, sizeof(float *));
			for(int j = 0; j < maxDetections; ++j) probs[j] = (float*)calloc(nClasses + 1, sizeof(float *));

			printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TOPIC: %s \r\n",topic_name.c_str());
		};

	~MyNode() {
		free(boxes);
		free(probs);
		for(int j = 0; j < maxDetections; ++j) free(probs[j]);
	}
	;


	void onImage(const sensor_msgs::ImageConstPtr& msg) {
		printf("Yolo: image received \r\n");
		if(readyToPublish==1)
		{
			readyToPublish = 0;
			 
			cv_bridge::CvImagePtr cv_ptr;
			try {
				cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
			} catch (cv_bridge::Exception& e) {
				ROS_ERROR("cv_bridge exception: %s", e.what());
				return;
			}

			// Convert to Darknet image format.
			image im = OpencvMat2DarkNetImage(cv_ptr->image);
			execute_yolo_model2(im, threshold,boxes, probs); // Returns bounding boxes and probabilities.

			publish_detections(cv_ptr->image, maxDetections, threshold, boxes, probs,names); 

			free_image(im);
			readyToPublish = 1;
		}
	}

	// Roughly the same as ipl_to_image() in image.c
	image OpencvMat2DarkNetImage(Mat src)
	{
	    unsigned char *data = (unsigned char *)src.data;
	    int h = src.rows;
	    int w = src.cols;
	    int c = src.channels();
	    int step = src.step1();
	    image out = make_image(w, h, c);
	    int i, j, k, count=0;;

	    for(k= 0; k < c; ++k){
	    //for(k= c-1; k >= 0; --k){
		for(i = 0; i < h; ++i){
		    for(j = 0; j < w; ++j){
		        out.data[count++] = float(data[i*step + j*c + k]/255.);
		    }
		}
	    }
	    return out;
	}
	Mat publish_detections(Mat img, int num, float thresh, box *boxesIn, float **probsIn, char **names)
	{
		int i;
		int cDetections = 0;
		box_prob* detections = (box_prob*)calloc(maxDetections, sizeof(box_prob));
		//printf("Number of bounding boxes %i: \n", num);
		for(i = 0; i < num; ++i){
			int topClass = max_index(probs[i],nClasses);
			float prob = probs[i][topClass];
			if(prob > thresh){
				int width = pow(prob, 1./2.)*10+1; // line thickness
				box b = boxes[i];
				 Scalar useColor(0, 0, 0);
				float x  = (b.x-b.w/2.)*(float)(img.cols);
				float y = (b.y-b.h/2.)*(float)(img.rows);
				float w   = b.w*(float)(img.cols);
				float h   = b.h*(float)(img.rows);
				printf("bb: %f %f %f %f \n", x,y,w,h);

				if(visualizeDetections){
					rectangle(img, Rect(x,y,w,h), useColor, 2, 8, 0);
					char numstr[30];
					sprintf(numstr, "%s %.2f",names[topClass], prob); 
					putText(img, numstr, Point(x+4,y-14+h),FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, 8, false);
				}

				// Detection in x and y coordinates (with x, y as upper left corner)
				detections[cDetections].x = x;
				detections[cDetections].y = y;
				detections[cDetections].w = w;
				detections[cDetections].h = h;
				detections[cDetections].prob = prob;
				detections[cDetections].objectType = topClass;

				cDetections++;
			}
		}


		/* Creating visual marker
		visualization_msgs::Marker marker;
		marker.header.frame_id = "/laser";
		marker.header.stamp = ros::Time();
		marker.ns = "my_namespace";
		marker.id = 0;
		marker.type = visualization_msgs::Marker::CYLINDER;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;*/

		
		// An estimate of the distance to the object is calculated using the camera setup.
		// Estimate is based on two assumptions: 1) The surface is flat. 2) The bottom of the bounding box is the bottom of the detected object.
		/*if(1){
			printf("Start bboxSAFE \n");
			double resolutionVertical = img.rows;
			double resolutionHorisontal = img.cols;
			//htf_safe_msgs::SAFEObstacleMsg msgObstacle;
			// MAYBE CLEARING IS NEEDED
			msgObstacle.xCoordinate.clear();
			msgObstacle.yCoordinate.clear();
			msgObstacle.zCoordinate.clear();
			msgObstacle.quality.clear();
			msgObstacle.objectType.clear();
			msgObstacle.objectID.clear();

			msgObstacle.header.stamp = ros::Time::now();

			for (int n = 0; n < cDetections;n++){
				double buttomRowPosition = detections[n].y+detections[n].h; // bbs(n,2)-bbs(n,4);
				double ColPosition = detections[n].x+detections[n].w/2; // bbs(n,2)-bbs(n,4);

				double distance = tan(M_PI/2-(angleTiltRad+FOV_verticalRad/2) + FOV_verticalRad*(resolutionVertical-buttomRowPosition)/resolutionVertical)*cameraHeight;
				double angle =((ColPosition-resolutionHorisontal/2)/resolutionHorisontal)*FOV_verticalRad;
				double xCoordinate = cos(angle)*distance;
				double yCoordinate = sin(angle)*distance;
				msgObstacle.xCoordinate.push_back(xCoordinate);
				msgObstacle.yCoordinate.push_back(yCoordinate);
				msgObstacle.zCoordinate.push_back(0.0);
				msgObstacle.quality.push_back(detections[n].prob);
				msgObstacle.objectType.push_back(detections[n].objectType);
				msgObstacle.objectID.push_back(0);
				//cout << "x1:" << bbs[n].x1 << ", y2:" << bbs[n].y2 << ", w3:" << bbs[n].width3 << ", h4:" << bbs[n].height4 << ", s5: " << bbs[n].score5 << ",a5: " << bbs[n].angle << endl;
				//cout << "Distance: " <<  bbs[n].distance << endl;
			}
			pub_bbSAFE.publish(msgObstacle);
		}*/


		// Create bounding box publisher (multi array)
		std_msgs::Float64MultiArray bboxMsg;
		bboxMsg.data.clear();

		for (int iBbs = 0; iBbs < cDetections; ++iBbs) {

			bboxMsg.data.push_back(detections[iBbs].x/img.cols);
			bboxMsg.data.push_back(detections[iBbs].y/img.rows);
			bboxMsg.data.push_back(detections[iBbs].w/img.cols);
			bboxMsg.data.push_back(detections[iBbs].h/img.rows);
			bboxMsg.data.push_back(detections[iBbs].prob);
			if(useRemapping){ 
				bboxMsg.data.push_back(remapYolo2NewObjectTypes[int(detections[iBbs].objectType)]);
			}
			else{
				bboxMsg.data.push_back(int(detections[iBbs].objectType));
			}
		}
		pub_bb.publish(bboxMsg);

		// Create image publisher showing yolo detections.
		if(visualizeDetections){

			//sensor_msgs::CameraInfoPtr cc(new sensor_msgs::CameraInfo(cinfor_->getCameraInfo()));
			sensor_msgs::ImagePtr msg_image_out = cv_bridge::CvImage(std_msgs::Header(),"bgr8", img).toImageMsg();
			msg_image_out->header.stamp = ros::Time::now();
			//pub_image.publish(msg_image_out, cc);
			pub_image.publish(msg_image_out);
			//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
			//imshow( "Display window", img);
		}
		free (detections);
		return img;
	}
private:
	double imageResize;
	float threshold;
	std::string basedir;
	std::string model_cfg;
	std::string weightfile;
	std::string topic_name;
	std::string datafile;
	bool visualizeDetections;

	cv::Mat img;
	ros::NodeHandle nh;
	image_transport::ImageTransport it;
	//image_transport::CameraPublisher pub_image;
	//image_transport::CameraSubscriber sub_image;
	image_transport::Publisher pub_image;
	image_transport::Subscriber sub_image;
	ros::Publisher pub_bb;
	ros::Publisher pub_bbSAFE;
	//boost::shared_ptr<camera_info_manager::CameraInfoManager> cinfor_;
	bool readyToPublish;
	image **alphabet;
	box *boxes;
	float **probs;
	char ** names; // Name of all classes. 
	bool useRemapping;
	list *options; 
	
	int maxDetections;
	int nClasses;
	double FOV_verticalDeg,FOV_horizontalDeg,angleTiltDegrees,cameraHeight;
	double FOV_verticalRad, FOV_horizontalRad,angleTiltRad;
};


int main(int argc, char** argv) {

	ros::init(argc, argv, "darknet_stuff");

	MyNode node;

	ros::spin();
}

/*int main()
{
	char cfgfile[] = "/home/repete/DeepLearningStuff/darknet/cfg/yolo-small.cfg";
	char weightfile[] = "/home/repete/DeepLearningStuff/darknet/weights/yolo-small.weights";
	char filename[] = "/home/repete/DeepLearningStuff/darknet/data/dog.jpg";

	printf("State0");
	float thresh = 0.2;
	load_yolo_model(cfgfile, weightfile);
	printf("State1");
	execute_yolo_model(filename, thresh);
	printf("State2");
	return 0;
}*/
