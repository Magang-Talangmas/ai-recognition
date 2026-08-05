#ifndef ATTENDANCE_VIDEO_CAPTURE_H
#define ATTENDANCE_VIDEO_CAPTURE_H

#include <stdbool.h>
#include <stdint.h>
#include "attendance/face_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VideoCapture VideoCapture;

/*
 * Open video capture stream (RTSP URL, webcam index, or video file)
 */
VideoCapture *video_capture_open(const char *source, int target_width, int target_height);

/*
 * Read next frame and any detected faces from stream.
 * Populates out_frame (BGR24 buffer) and out_faces (detected face structures & embeddings).
 * Returns true on success.
 */
bool video_capture_read_frame(
    VideoCapture *cap,
    ImageBuffer *out_frame,
    FaceResult *out_faces,
    int max_faces,
    int *num_faces_out,
    double now_time
);

/*
 * Close video capture stream and free resources.
 */
void video_capture_close(VideoCapture *cap);

#ifdef __cplusplus
}
#endif

#endif /* ATTENDANCE_VIDEO_CAPTURE_H */
