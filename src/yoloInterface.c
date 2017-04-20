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

//char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
//image voc_labels[20];
network glo_net;



/*void execute_yolo_model(image im, float thresh) //, *float boxesOut, *float probsOut
{
    clock_t time1,time2;
    time1=clock();
    detection_layer l = glo_net.layers[glo_net.n-1];
    srand(2222222);
    
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    image sized = resize_image(im, glo_net.w, glo_net.h);
    float *X = sized.data;
    time2=clock();
    float *predictions = network_predict(glo_net, X);
    printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time1));
    printf("Not related to network %f seconds.\n", sec(time2-time1));
    convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
    if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
    draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, 20);
    show_image(im, "predictions");
    free_image(im);
#ifdef OPENCV
    cvWaitKey(1);
#endif

}*/

/*void execute_yolo_model(image im, float thresh,box *boxes,float **probs) //, *float boxesOut, *float probsOut
{
    clock_t time1,time2;
    time1=clock();
    detection_layer l = glo_net.layers[glo_net.n-1];
    srand(2222222);
    
    int j;
    float nms=.5;
    //box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    //float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    //
    image sized = resize_image(im, glo_net.w, glo_net.h);
    float *X = sized.data;
    time2=clock();
    float *predictions = network_predict(glo_net, X);
    printf("Predicted in %f seconds.\n", sec(clock()-time1));
    printf("Not related to network %f seconds.\n", sec(time2-time1));
    convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
    if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
    free_image(sized);
}*/

void execute_yolo_model2(image im, float thresh,box *boxes,float **probs) //, *float boxesOut, *float probsOut
{
    printf("test %i \n", 0);
    clock_t time1,time2;
    time1=clock();
    printf("test %i \n", 1);
    //detection_layer l1 = glo_net.layers[glo_net.n-1];
    layer l1 = glo_net.layers[glo_net.n-1];
	
    srand(2222222);
    
    int j;
    float nms=.5;
    //box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    //float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    //
    printf("test %i \n", 2);
    image imresized = resize_image(im, glo_net.w, glo_net.h);
    float *X = imresized.data;
    time2=clock();
    printf("test %i \n", 3);
    //float *predictions = network_predict(glo_net, X);
    network_predict(glo_net, X);

    printf("Predicted in %f seconds.\n", sec(clock()-time1));
    printf("Not related to network %f seconds.\n", sec(time2-time1));
    //convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
    float hier_thresh = 0; // ???????

    //printf("Inputs to get_region_boxes: \n");
    //printf("imresized.w: %i \n",imresized.w);
    //printf("imresized.h: %i \n",imresized.h);
    //printf("glo_net.w: %i \n",glo_net.w);
    //printf("glo_net.h: %i \n",glo_net.h);
    get_region_boxes(l1, imresized.w, imresized.h, glo_net.w, glo_net.h, thresh, probs, boxes, 0, 0, hier_thresh, 1);
    printf("test %i \n", 4);
    if (nms) do_nms_obj(boxes, probs, l1.w*l1.h*l1.n, l1.classes, nms);

    printf("test %i \n", 5);
    //if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
    free_image(imresized);
}

int load_yolo_model(char *cfgfile, char *weightfile)//, int& n_classes)
{
    int maxSize = 0;
    printf("Loading: %s \r\n",cfgfile);
	//printf("Test0 \r\n");
    glo_net = parse_network_cfg(cfgfile);
	//printf("Test1, %s %s \r\n",cfgfile,weightfile);
    if(weightfile){
        load_weights(&glo_net, weightfile);
    }
	//printf("Test2 \r\n");
    set_batch_network(&glo_net, 1);
	//printf("Test3 \r\n");
    //detection_layer l = glo_net.layers[glo_net.n-1];
    layer l = glo_net.layers[glo_net.n-1];
	//printf("Test4 \r\n");
	//printf("Done loading %s \r\n",cfgfile);

    //max_detections = l.side*l.side*l.n;
    //n_classes = l.classes;
    //printf("Loading: max_detection %i and classes %i \r\n",l.side*l.side*l.n,l.classes);

    return l.w*l.h*l.n; // returns the maximum possible number of detections
}

int get_nclasses(){
	layer l = glo_net.layers[glo_net.n-1];
	return l.classes;
}

/*void execute_yolo_model_file(char *filename, float thresh) //, *float boxesOut, *float probsOut
{
    detection_layer l = glo_net.layers[glo_net.n-1];
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, glo_net.w, glo_net.h);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(glo_net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
        draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, 0, 20);
        show_image(im, "predictions");
        save_image(im, "predictions");

        show_image(sized, "resized");
        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}*/
