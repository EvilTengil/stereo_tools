#include "cv.h"
#include "highgui.h"

#include "stdio.h"

int main(int argc, char** argv) 
{

    if (argc != 3) return 1;

    int cam1 = atoi(argv[1]);
    int cam2 = atoi(argv[2]);

    CvCapture* capture1 = 0;
    CvCapture* capture2 = 0;
    IplImage *frame1 = 0;
    IplImage *frame2 = 0;
    int img_num = 0;

    if (!(capture1 = cvCaptureFromCAM(cam1)))
        printf("Cannot initialize camera 0\n");

    cvSetCaptureProperty( capture1, CV_CAP_PROP_FRAME_WIDTH, 640 );
    cvSetCaptureProperty( capture1, CV_CAP_PROP_FRAME_HEIGHT, 480 );


    if (!(capture2 = cvCaptureFromCAM(cam2)))
        printf("Cannot initialize camera 1\n");

    cvSetCaptureProperty( capture2, CV_CAP_PROP_FRAME_WIDTH, 640 );
    cvSetCaptureProperty( capture2, CV_CAP_PROP_FRAME_HEIGHT, 480 );

    cvNamedWindow("Capture0", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Capture1", CV_WINDOW_AUTOSIZE);

    bool done = false;
    while (!done) {
        
        frame1 = cvQueryFrame(capture1);
        frame2 = cvQueryFrame(capture2);

        if (!frame1 || !frame2)
            break;
                        

        cvShowImage("Capture0", frame1); // Display the frame
        cvShowImage("Capture1", frame2); // Display the frame
        
	int key = cvWaitKey(10);
	    
        if (key == 27) {
            done = true;
	}
        
        if (key == 32) {
            char filename[256];
            sprintf(filename, "%04d_left.png", img_num);
            cvSaveImage(filename, frame1, 0); // Save this image
            sprintf(filename, "%04d_right.png", img_num);
            cvSaveImage(filename, frame2, 0); // Save this image
	    ++img_num;
	}
    }
    
    cvReleaseImage(&frame1);
    cvReleaseImage(&frame2);
    cvReleaseCapture(&capture1);
    cvReleaseCapture(&capture2);

    return 0;
}
