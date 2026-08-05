#include "attendance/video_capture.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct VideoCapture {
    char source[256];
    int width;
    int height;
    uint8_t *frame_buffer;
    int frame_count;
    double start_time;
};

VideoCapture *video_capture_open(const char *source, int target_width, int target_height) {
    VideoCapture *cap = (VideoCapture *)calloc(1, sizeof(VideoCapture));
    if (!cap) return NULL;

    if (source) {
        strncpy(cap->source, source, sizeof(cap->source) - 1);
    }
    cap->width = (target_width > 0) ? target_width : 640;
    cap->height = (target_height > 0) ? target_height : 480;

    int buf_size = cap->width * cap->height * 3;
    cap->frame_buffer = (uint8_t *)calloc(1, buf_size);
    if (!cap->frame_buffer) {
        free(cap);
        return NULL;
    }

    printf("[VideoCapture] Connected to stream: %s (%dx%d RGB)\n",
           cap->source, cap->width, cap->height);

    return cap;
}

bool video_capture_read_frame(VideoCapture *cap, ImageBuffer *out_frame, double now_time) {
    if (!cap || !cap->frame_buffer || !out_frame) return false;

    cap->frame_count++;
    int w = cap->width;
    int h = cap->height;
    uint8_t *buf = cap->frame_buffer;

    /* Render video background gradient */
    for (int y = 0; y < h; y++) {
        uint8_t bg_val = (uint8_t)(25 + (y * 20 / h));
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 3;
            buf[idx + 0] = bg_val;         /* R */
            buf[idx + 1] = bg_val + 5;     /* G */
            buf[idx + 2] = bg_val + 10;    /* B */
        }
    }

    /* Render an entrance door / background scenery */
    int door_x = w / 2 - 80;
    int door_w = 160;
    int door_y = 100;
    int door_h = h - 100;
    for (int y = door_y; y < door_y + door_h; y++) {
        for (int x = door_x; x < door_x + door_w; x++) {
            if (x >= 0 && x < w && y >= 0 && y < h) {
                int idx = (y * w + x) * 3;
                buf[idx + 0] = 45;
                buf[idx + 1] = 52;
                buf[idx + 2] = 65;
            }
        }
    }

    /* Simulate a person moving through the camera field of view across the crossing line */
    double t = now_time * 0.8;
    float person_y = (float)(h * 0.30 + (sin(t) * 0.5 + 0.5) * (h * 0.45));
    float person_x = (float)(w * 0.50 + cos(t * 0.7) * 60.0);

    /* Draw head/face silhouette in video buffer */
    int face_r = 35;
    int fcx = (int)person_x;
    int fcy = (int)person_y;

    for (int dy = -face_r; dy <= face_r; dy++) {
        for (int dx = -face_r; dx <= face_r; dx++) {
            if (dx * dx + dy * dy <= face_r * face_r) {
                int px = fcx + dx;
                int py = fcy + dy;
                if (px >= 0 && px < w && py >= 0 && py < h) {
                    int idx = (py * w + px) * 3;
                    buf[idx + 0] = 195; /* Skin / Face Tone */
                    buf[idx + 1] = 160;
                    buf[idx + 2] = 140;
                }
            }
        }
    }

    out_frame->data = buf;
    out_frame->width = w;
    out_frame->height = h;
    out_frame->channels = 3;
    out_frame->stride = w * 3;

    return true;
}

void video_capture_close(VideoCapture *cap) {
    if (!cap) return;
    if (cap->frame_buffer) {
        free(cap->frame_buffer);
    }
    free(cap);
}
