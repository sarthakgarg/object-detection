
#include <stdio.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/bgsegm.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv_core247d.lib")
#pragma comment(lib, "opencv_imgproc247d.lib")   //MAT processing
#pragma comment(lib, "opencv_objdetect247d.lib") //HOGDescriptor
//#pragma comment(lib, "opencv_gpu247d.lib")
//#pragma comment(lib, "opencv_features2d247d.lib")
#pragma comment(lib, "opencv_highgui247d.lib")
#pragma comment(lib, "opencv_ml247d.lib")
//#pragma comment(lib, "opencv_stitching247d.lib");
//#pragma comment(lib, "opencv_nonfree247d.lib");
#pragma comment(lib, "opencv_video247d.lib")
#else
#pragma comment(lib, "opencv_core247.lib")
#pragma comment(lib, "opencv_imgproc247.lib")
#pragma comment(lib, "opencv_objdetect247.lib")
//#pragma comment(lib, "opencv_gpu247.lib")
//#pragma comment(lib, "opencv_features2d247.lib")
#pragma comment(lib, "opencv_highgui247.lib")
#pragma comment(lib, "opencv_ml247.lib")
//#pragma comment(lib, "opencv_stitching247.lib");
//#pragma comment(lib, "opencv_nonfree247.lib");
#pragma comment(lib, "opencv_video247d.lib")
#endif

using namespace cv;
using namespace std;
using namespace bgsegm;


int main(int argc, char* argv[])
{


  /* Check for the input parameter correctness. */
  if (argc != 2) {
    cerr <<"Incorrect input : Video Filename required" << endl;
    cerr <<"exiting..." << endl;
    return 1;
  }


 //global variables
 Mat frame; //current frame
 Mat resize_blur_Img;
 Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
 Mat binaryImg, morphPartial;
 //Mat TestImg;
 Mat ContourImg; //fg mask fg mask generated by MOG2 method
 Ptr< BackgroundSubtractor> pMOG2; //MOG2 Background subtractor

 pMOG2 = createBackgroundSubtractorMOG2(300,32,true);//(300,32,true);//300,0.0);


 //char fileName[100] = "datasample1.mov"; //video\\mm2.avi"; //mm2.avi"; //cctv 2.mov"; //mm2.avi"; //";//_p1.avi";
 VideoCapture stream1(argv[1]);   //0 is the id of video device.0 if you have only one camera

 //morphology element
 Mat element = getStructuringElement(MORPH_RECT, Size(7, 7), Point(3,3) );

 //unconditional loop
 while (true) {
  Mat cameraFrame;
  if(!(stream1.read(frame))) //get one frame form video
   break;

  //Resize
  resize(frame, resize_blur_Img, Size(frame.size().width/3, frame.size().height/3) );
  //Blur
  blur(resize_blur_Img, resize_blur_Img, Size(2, 2) );
  //Background subtraction
  pMOG2->apply(resize_blur_Img, fgMaskMOG2, -0.5);//,-0.5);

  ///////////////////////////////////////////////////////////////////
  //pre procesing
  //1 point delete
  //morphologyEx(fgMaskMOG2, fgMaskMOG2, CV_MOP_ERODE, element);
  morphologyEx(fgMaskMOG2, morphPartial, CV_MOP_CLOSE, element);
  //morphologyEx(fgMaskMOG2, testImg, CV_MOP_OPEN, element);
  //Shadow delete
  //Binary
  threshold(morphPartial, binaryImg, 128, 255, CV_THRESH_BINARY);

  //Find contour
  ContourImg = binaryImg.clone();
  //less blob delete
  vector< vector< Point> > contours;
  findContours(ContourImg,
            contours, // a vector of contours
            CV_RETR_EXTERNAL, // retrieve the external contours
            CV_CHAIN_APPROX_NONE); // all pixels of each contours

  vector< Rect > output;
  vector< vector< Point> >::iterator itc= contours.begin();
  while (itc!=contours.end()) {

   //Create bounding rect of object
   //rect draw on origin image
   Rect mr= boundingRect(Mat(*itc));
      if(mr.area() > 15000)
     rectangle(resize_blur_Img, mr, CV_RGB(255,0,0));
   ++itc;
  }


  ///////////////////////////////////////////////////////////////////

  //Display
  //namedWindow("cv_Shadow_Removed", WINDOW_NORMAL);
  namedWindow("cv_Blur_Resize", WINDOW_OPENGL);
  //namedWindow("cv_MOG2", WINDOW_NORMAL);

  //  imshow("cv_Shadow_Removed", binaryImg);
  imshow("cv_Blur_Resize", resize_blur_Img);
  //imshow("cv_MOG2", fgMaskMOG2);


   if (waitKey(2) == 'q' )
    break;
 }

}