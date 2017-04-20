#ifndef YOLO_H
#define YOLO_H
#include "image.h"

typedef struct{
    float x, y, w, h,prob,objectType;
} box_prob;

//void execute_yolo_model(image im, float thresh);
void execute_yolo_model(image im, float thresh,box *boxes,float **probs);
void execute_yolo_model2(image im, float thresh,box *boxes,float **probs);
//int load_yolo_model(char *cfgfile, char *weightfile);

//void load_yolo_model(char *cfgfile, char *weightfile, int& max_detections, int& n_classes);
void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);
int get_nclasses();
int load_yolo_model(char *cfgfile, char *weightfile);
#endif

