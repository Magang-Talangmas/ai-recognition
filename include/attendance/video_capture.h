#ifndef ATTENDANCE_VIDEO_CAPTURE_H
#define ATTENDANCE_VIDEO_CAPTURE_H

#include <stdbool.h>
#include <stdint.h>
#include "attendance/face_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VideoCapture VideoCapture;

/* Initialize camera or RTSP stream video capture */
VideoCapture *video_capture_open(const char *source, int target_width, int target_height);

/* Read latest frame. Returns true if frame was read successfully */
bool video_capture_read_frame(VideoCapture *cap, ImageBuffer *out_frame, double now_time);

/* Release video capture resources */
void video_capture_close(VideoCapture *cap);

#ifdef __cplusplus
}
#endif

#endif /* ATTENDANCE_VIDEO_CAPTURE_H */
