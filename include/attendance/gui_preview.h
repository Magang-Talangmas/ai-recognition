#ifndef ATTENDANCE_GUI_PREVIEW_H
#define ATTENDANCE_GUI_PREVIEW_H

#include <stdbool.h>
#include <stdint.h>
#include "attendance/face_engine.h"
#include "attendance/vision_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GuiWindow GuiWindow;

/* Create and open a native Windows preview window for the camera stream */
GuiWindow *gui_window_create(const char *title, int width, int height);

/* Destroy window and cleanup GUI resources */
void gui_window_destroy(GuiWindow *win);

/* Process Windows events (keys, close button). Returns false if window was closed or 'q'/ESC pressed */
bool gui_window_process_events(GuiWindow *win);

/* Render frame with bounding boxes, HUD overlay, and line crossing to window */
void gui_window_render(
    GuiWindow *win,
    const ImageBuffer *frame,
    const FaceResult *faces,
    int num_faces,
    const CentroidTracker *tracker,
    float line_y_ratio,
    float display_fps,
    float inference_ms,
    const char *last_event_msg
);

#ifdef __cplusplus
}
#endif

#endif /* ATTENDANCE_GUI_PREVIEW_H */
