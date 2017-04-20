#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "cv.h"
#endif

network glo_net;

void execute_yolo_model2(image im, float thresh,box *boxes,float **probs) //, *float boxesOut, *float probsOut
{


    clock_t time1,time2;
    time1=clock();

    layer l1 = glo_net.layers[glo_net.n-1];
	
    srand(2222222);
    
    int j;
    float nms=.5;

    image imresized = resize_image(im, glo_net.w, glo_net.h);
    float *X = imresized.data;
    time2=clock();


    network_predict(glo_net, X);

    printf("Predicted in %f seconds.\n", sec(clock()-time1));
    printf("Not related to network %f seconds.\n", sec(time2-time1));

    float hier_thresh = 0; // ???????


    get_region_boxes(l1, imresized.w, imresized.h, glo_net.w, glo_net.h, thresh, probs, boxes, 0, 0, hier_thresh, 1);

    if (nms) do_nms_obj(boxes, probs, l1.w*l1.h*l1.n, l1.classes, nms);

    free_image(imresized);
}

int load_yolo_model(char *cfgfile, char *weightfile)
{
    int maxSize = 0;
    printf("Loading: %s \r\n",cfgfile);

    glo_net = parse_network_cfg(cfgfile);

    if(weightfile){
        load_weights(&glo_net, weightfile);
    }

    set_batch_network(&glo_net, 1);


    layer l = glo_net.layers[glo_net.n-1];

    return l.w*l.h*l.n; // returns the maximum possible number of detections
}

int get_nclasses(){
	layer l = glo_net.layers[glo_net.n-1];
	return l.classes;
}

