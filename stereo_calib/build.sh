gcc stereo_calib.cpp -o stereo_calib \
  -I/usr/local/include/opencv \
  -I/usr/local/include \
  -L/usr/local/lib \
  -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann  
