#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/opengl.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm/glm.hpp>
#include <glm/glm/gtc/matrix_transform.hpp>
#include <glm/glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "./stb_image.h"
#include "./Shader.h"
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

void loadMatToOpenGL(Mat& image,Shader& textureShader,Shader& cubeShader,Point2f& center){
  glPixelStorei(GL_UNPACK_ALIGNMENT, (image.step & 3) ? 1 : 4); //Remove openCV padding
  if(image.empty()){
      std::cout << "image empty" << std::endl;
  }else{
      
      //Vertices for background
      float vertices[] = {
        // positions          // colors           // texture coords
        1.0f,  1.0f, 1.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
        1.0f, -1.0f, 1.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 1.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 1.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left 
      };
      unsigned int indices[] = {  
            0, 1, 3, // first triangle
            1, 2, 3  // second triangle
      };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    
    float cubeVertices[] = {
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
    };
    // world space positions of our cubes --------------------- Set with input
    glm::vec3 cubePosition = glm::vec3(center.x,  center.y,  0.0f);
    
    
    unsigned int cVBO, cVAO;
    glGenVertexArrays(1, &cVAO);
    glGenBuffers(1, &cVBO);

    glBindVertexArray(cVAO);

    glBindBuffer(GL_ARRAY_BUFFER, cVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    
    cv::flip(image, image, 0); //Flip image as openCV read up-side-down.
    unsigned int texture,texture2;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Set texture clamping method
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);


    glTexImage2D(GL_TEXTURE_2D,     // Type of texture
                    0,                 // Pyramid level (for mip-mapping) - 0 is the top level
                    GL_RGB,            // Internal colour format to convert to
                    image.cols,          // Image width  i.e. 640 for Kinect in standard mode
                    image.rows,          // Image height i.e. 480 for Kinect in standard mode
                    0,                 // Border width in pixels (can either be 1 or 0)
                    GL_BGR, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
                    GL_UNSIGNED_BYTE,  // Image data type
                    image.ptr());        // The actual image data itself

    glGenerateMipmap(GL_TEXTURE_2D); //Could maybe remove as we dont need mipmaps for background
    
    //----------------------------------BOX----------------------------------
    glGenTextures(1, &texture2);
    glBindTexture(GL_TEXTURE_2D, texture2);
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    int width, height, nrChannels;
    auto data = stbi_load("../container.jpg", &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    
    
    textureShader.use();
    textureShader.setInt("ourTexture", 0);
    
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // bind Texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    textureShader.use();
    
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    // render scene
    glEnable(GL_DEPTH_TEST);
    cubeShader.use();
    cubeShader.setInt("texture1", 1);
    
    glm::mat4 view          = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
    glm::mat4 projection    = glm::mat4(1.0f);
    projection = glm::perspective(glm::radians(45.0f), (float)image.cols / (float)image.rows, 0.1f, 100.0f);
    view       = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));
    
    cubeShader.setMat4("projection", projection); // note: currently we set the projection matrix each frame, but since the projection matrix rarely changes it's often best practice to set it outside the main loop only once.
    cubeShader.setMat4("view", view);
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture2);
    glBindVertexArray(cVAO);
    
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::scale(model,glm::vec3(0.2f, 0.2f, 0.2f));
    model = glm::translate(model, cubePosition);
    float angle = 20.0f;
    model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
    cubeShader.setMat4("model", model);

    glDrawArrays(GL_TRIANGLES, 0, 36);
    glDisable(GL_DEPTH_TEST);
  }
}

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

Point compute2DPolygonCentroid(const Point* vertices, int vertexCount)
{
    Point centroid = {0, 0};
    double signedArea = 0.0;
    double x0 = 0.0; // Current vertex X
    double y0 = 0.0; // Current vertex Y
    double x1 = 0.0; // Next vertex X
    double y1 = 0.0; // Next vertex Y
    double a = 0.0;  // Partial signed area

    int lastdex = vertexCount-1;
    const Point* prev = &(vertices[lastdex]);
    const Point* next;

    // For all vertices in a loop
    for (int i=0; i<vertexCount; ++i)
    {
        next = &(vertices[i]);
        x0 = prev->x;
        y0 = prev->y;
        x1 = next->x;
        y1 = next->y;
        a = x0*y1 - x1*y0;
        signedArea += a;
        centroid.x += (x0 + x1)*a;
        centroid.y += (y0 + y1)*a;
        prev = next;
    }

    signedArea *= 0.5;
    centroid.x /= (6.0*signedArea);
    centroid.y /= (6.0*signedArea);

    return centroid;
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
    int scale = 60;
    
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(w, h, "Augmented Reality", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    Shader textureShader("../textureShader.vs", "../textureShader.fs");
    Shader cubeShader("../cubeShader.vs", "../cubeShader.fs");

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
	        //tracker->init(img, bbox);
            cout << "Initialized tracker" << endl;
            
            Mat res1,real,invreal,oldMask;
            vector<Mat> Rs,Ts,Ns;
            vector<int> sols;
            bool init = 0;
            while (cap.read(img)) {
                //bool ok = tracker->update(img, bbox);
                if (1) {
                    //rectangle(img, bbox, Scalar( 255, 0, 0 ), 2, 1 );
                    //Mat temp = img(bbox);
                    Mat tempThresh = preProcessing(img,180,255,3); //(temp,130,255,5); //New (temp,80,255,6)
                    imshow("Smallthresh", tempThresh);
                    initialPoints = getContours(tempThresh);
//                     for (auto& p: initialPoints){ //Transform back corner points from mask to real image.
//                         p.x += bbox.x;
//                         p.y += bbox.y;
//                     }
                }
                auto mask = img.clone();
                Point2f centerPoint{};
                if (initialPoints.size() == 4) {
                    centerPoint = compute2DPolygonCentroid(&initialPoints[0],4);
                    cout << "Centerpoint: (" << centerPoint.x << "," << centerPoint.y << ")" << endl;
                    centerPoint.x /= w;
                    centerPoint.y /= h;
                    centerPoint.x -= 0.5;
                    centerPoint.y -= 0.5;
                    centerPoint *= 2;
                    cout << "Centerpoint CLIPSPACE : (" << centerPoint.x << "," << centerPoint.y << ")" << endl;
                    init = 1;
                    reorder(initialPoints);
                    auto homo = getWarp(mask,imgWarp,initialPoints,w,h); //mask will contain RGB-transformed homography mask
                    //Decompose homography.
                    decomposeHomographyMat(homo,K,Rs,Ts,Ns);
                    vector<Point2f> floatPts;
                    Mat(initialPoints).convertTo(floatPts,Mat(floatPts).type());
                    filterHomographyDecompByVisibleRefpoints(Rs,Ns,imgPts,floatPts,sols);
                    oldMask = mask;
                    threshold(mask,real,1,255,THRESH_BINARY);
                    bitwise_not(real,invreal);
                    bitwise_and(invreal,img,res1);
                    add(res1,mask,img);
                    //Transform from tempImg-coordinates to img-coordinates. DONE!
                    //Resize lambo picture properly. DONE?
                    //Overlay images properly. DONE!
                    //Later, move out warpcode + return homography so that old position gets displayed in case of failed new detection DONE!
                    //Adjust parameters to detect more borders inside bbox <-- Add parameters to getContours/preProcessing! DONE?
                    //Fix reorder giving same point DONE!
                    //Fix reorder so that we cant have a homography from one side to the other DONE!
                    //Decompose homography DONE!
                    //Select correct solution HALF-DONE, Normal will be correct
                    //Insert 3D-model HALF-DONE! Add points together, divide by 4 to get center and send center to openGL
                } else {
                    // 4 points could not be detected - Use previous mask in new frame
                    if (init) {
                        bitwise_and(invreal,img,res1);
                        add(res1,oldMask,img);
                    }
                }
                //imshow("Tracking", img);
                loadMatToOpenGL(img,textureShader,cubeShader,centerPoint);
                glfwSwapBuffers(window);
                glfwPollEvents();
                waitKey(1);
            }
        }
    }
    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}
