#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int tmax = 255,tmin = 0, kernsize = 3;

    
vector<Point> getContours(Mat& imgDil){
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    
    findContours(imgDil,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> conPoly(contours.size());
    string objectType;
    vector<Point> biggest;
    double maxArea = 0;
    for (int i = 0; i < contours.size(); i++){
        auto area = contourArea(contours[i]);
        if (area > 100000) {
            auto peri = arcLength(contours[i],1);
            approxPolyDP(contours[i],conPoly[i],0.02*peri,1);
            if (area > maxArea && conPoly[i].size() == 4){
                //drawContours(img,conPoly,i,Scalar(255,0,255),5);
                biggest = {{conPoly[i][0],conPoly[i][1],conPoly[i][2],conPoly[i][3]}};
                maxArea = area;
            }
        }
    }
    cout<<  maxArea << endl;
    return biggest;
}

Mat preProcessing(Mat& img,int threshMin,int threshMax,int kernelSize){
    Mat imgGray,imgBlur,imgCanny,imgDia;
    cvtColor(img,imgGray,COLOR_BGR2GRAY);
    GaussianBlur(imgGray,imgBlur,Size(3,3),3,0);
    Canny(imgGray,imgCanny,threshMin,threshMax);
    Mat kernel = getStructuringElement(MORPH_RECT,Size(kernelSize,kernelSize));
    dilate(imgCanny,imgDia,kernel);
    return imgDia;
}

bool up (Point& p) {
    return p.y > 0 or (p.y == 0 and p.x >= 0);
}

void reorder(vector<Point>& initialPoints){
        for (int i = 0; i < 4; i++) {
        cout << "Before reorder = (" <<initialPoints[i].x << ","<< initialPoints[i].y << ")"<< endl;
    }
    
    //Fix case where 45deg gives points at different sides of image. std::rotate maybe
    sort(initialPoints.begin(), initialPoints.end(), [] (Point& a, Point& b) {
        return up(a) == up(b) ? a.x * b.y > a.y * b.x : up(a) < up(b);
    });
    //Below, hack so that homography doesnt snap. Would be good to incorporate into sort.
    if (initialPoints[2].x > initialPoints[3].x && initialPoints[2].y > initialPoints[3].y){
        swap(initialPoints[2],initialPoints[3]);
    }
    if (initialPoints[0].x > initialPoints[1].x && initialPoints[0].y > initialPoints[1].y){
        swap(initialPoints[0],initialPoints[1]);
    }
    if (initialPoints[2].x > initialPoints[1].x && initialPoints[2].y > initialPoints[1].y){
        swap(initialPoints[1],initialPoints[2]);
    }
        for (int i = 0; i < 4; i++) {
        cout << "After reorder = (" <<initialPoints[i].x << ","<< initialPoints[i].y << ")"<< endl;
    }
}

Mat getWarp(Mat& img,Mat& imgWarp,vector<Point>& pts,float w, float h) {
    Point2f src[4] = {{0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h}};
    Point2f dst[4] = {pts[0],pts[1],pts[2],pts[3]};
    auto homography = getPerspectiveTransform(src,dst);
    warpPerspective(imgWarp,img,homography,Point(w,h));
    return homography;
}


int main() {
    VideoCapture cap(2);
    Mat img,imgWarp;
    string path = "../Resources/lambo.png";
    imgWarp = imread(path);
    cap.read(img);
    resize(imgWarp,imgWarp,Size(),img.cols/imgWarp.cols,img.rows/imgWarp.rows,INTER_LINEAR);
    float w{static_cast<float>(img.cols)},h{static_cast<float>(img.rows)};
    Mat K = (Mat_<int>(3,3)<<1109,0,h/2,0,1109,w/2,0,0,1); //camera intrinsics
    vector<Point2f> imgPts = {{0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h}};
    int scale = 80;

    while(true) {
        cap.read(img);
        Mat imgThresh = preProcessing(img,180,255,3);
        auto initialPoints = getContours(imgThresh);
        imshow("Thresh", imgThresh);
        if (initialPoints.size() != 4) {
            waitKey(1);
            continue;
        } else {
            cout << "Found target" << endl;
            //Found target
            Ptr<TrackerMIL> tracker = TrackerMIL::create();
            auto bbox = boundingRect(initialPoints);
            cout << "Loaded tracker" << endl;
            bbox.width += scale;
            bbox.height += scale;
            if(bbox.x-scale/2 > 0) {
                bbox.x -= scale/2;
            }
            if(bbox.y-scale/2 > 0) {
                bbox.y -= scale/2;
            }
            
            cout << "Adjusted BBOX" << endl;
	        tracker->init(img, bbox);
            cout << "Initialized tracker" << endl;
            
            Mat res1,real,invreal,oldMask;
            vector<Mat> Rs,Ts,Ns;
            vector<int> sols;
            bool init = 0;
            while (cap.read(img)) {
                bool ok = tracker->update(img, bbox);
                if (ok) {
                    //rectangle(img, bbox, Scalar( 255, 0, 0 ), 2, 1 );
                    Mat temp = img(bbox);
                    Mat tempThresh = preProcessing(temp,80,255,6); //(temp,130,255,5); //New (temp,80,255,6)
                    imshow("Smallthresh", tempThresh);
                    initialPoints = getContours(tempThresh);
                    for (auto& p: initialPoints){ //Transform back corner points from tracked mask to real image.
                        p.x += bbox.x;
                        p.y += bbox.y;
                    }
                }
                auto mask = img.clone();
                if (initialPoints.size() == 4) {
                    init = 1;
                    reorder(initialPoints);
                    auto homo = getWarp(mask,imgWarp,initialPoints,w,h); //mask will contain RGB-transformed homography mask
                    oldMask = mask;
                    threshold(mask,real,1,255,THRESH_BINARY);
                    bitwise_not(real,invreal);
                    bitwise_and(invreal,img,res1);
                    add(res1,mask,img);
                } else {
                    // 4 points could not be detected - Use previous mask in new frame
                    if (init) {
                        bitwise_and(invreal,img,res1);
                        add(res1,oldMask,img);
                    }
                }
                imshow("Tracking", img);
                waitKey(1);
            }
        }
    }
    return 0;
}
