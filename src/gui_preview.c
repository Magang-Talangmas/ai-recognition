#include "attendance/gui_preview.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>

struct GuiWindow {
    HWND hwnd;
    HDC hdc;
    HDC mem_dc;
    HBITMAP mem_bitmap;
    HBITMAP old_bitmap;
    int width;
    int height;
    bool should_close;
    HFONT font_regular;
    HFONT font_bold;
    HFONT font_title;
};

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    GuiWindow *win = (GuiWindow *)GetWindowLongPtr(hwnd, GWLP_USERDATA);

    switch (msg) {
        case WM_CLOSE:
        case WM_DESTROY:
            if (win) win->should_close = true;
            PostQuitMessage(0);
            return 0;

        case WM_KEYDOWN:
            if (wparam == VK_ESCAPE || wparam == 'Q' || wparam == 'q') {
                if (win) win->should_close = true;
                PostQuitMessage(0);
                return 0;
            }
            break;

        case WM_ERASEBKGND:
            return 1; /* Suppress flicker on redraw */

        default:
            break;
    }
    return DefWindowProcA(hwnd, msg, wparam, lparam);
}

GuiWindow *gui_window_create(const char *title, int width, int height) {
    HINSTANCE hInstance = GetModuleHandle(NULL);

    const char *CLASS_NAME = "TalangmasCameraPreviewClass";
    WNDCLASSEXA wc = {0};
    wc.cbSize = sizeof(WNDCLASSEXA);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.lpszClassName = CLASS_NAME;

    RegisterClassExA(&wc);

    RECT rect = {0, 0, width, height};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW & ~WS_MAXIMIZEBOX & ~WS_THICKFRAME, FALSE);

    HWND hwnd = CreateWindowExA(
        WS_EX_APPWINDOW,
        CLASS_NAME,
        title ? title : "Talangmas AI Recognition - Camera Stream",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rect.right - rect.left,
        rect.bottom - rect.top,
        NULL, NULL, hInstance, NULL
    );

    if (!hwnd) {
        return NULL;
    }

    GuiWindow *win = (GuiWindow *)calloc(1, sizeof(GuiWindow));
    if (!win) {
        DestroyWindow(hwnd);
        return NULL;
    }

    win->hwnd = hwnd;
    win->width = width;
    win->height = height;
    win->should_close = false;

    SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)win);

    win->hdc = GetDC(hwnd);
    win->mem_dc = CreateCompatibleDC(win->hdc);
    win->mem_bitmap = CreateCompatibleBitmap(win->hdc, width, height);
    win->old_bitmap = (HBITMAP)SelectObject(win->mem_dc, win->mem_bitmap);

    win->font_regular = CreateFontA(14, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                   DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                   CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");

    win->font_bold = CreateFontA(14, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                                DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");

    win->font_title = CreateFontA(18, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                                 DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                 CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    return win;
}

void gui_window_destroy(GuiWindow *win) {
    if (!win) return;

    if (win->mem_dc) {
        SelectObject(win->mem_dc, win->old_bitmap);
        DeleteObject(win->mem_bitmap);
        DeleteDC(win->mem_dc);
    }

    if (win->font_regular) DeleteObject(win->font_regular);
    if (win->font_bold) DeleteObject(win->font_bold);
    if (win->font_title) DeleteObject(win->font_title);

    if (win->hdc && win->hwnd) {
        ReleaseDC(win->hwnd, win->hdc);
    }

    if (win->hwnd) {
        DestroyWindow(win->hwnd);
    }

    free(win);
}

bool gui_window_process_events(GuiWindow *win) {
    if (!win || win->should_close) return false;

    MSG msg;
    while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            win->should_close = true;
            return false;
        }
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }

    return !win->should_close;
}

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
) {
    if (!win || !win->hwnd || !win->mem_dc) return;

    HDC hdc = win->mem_dc;
    int w = win->width;
    int h = win->height;

    /* 1. Render Video Frame / Background */
    if (frame && frame->data && frame->width > 0 && frame->height > 0) {
        BITMAPINFO bmi = {0};
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = frame->width;
        bmi.bmiHeader.biHeight = -frame->height; /* Top-down DIB */
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = (frame->channels == 4) ? 32 : 24;
        bmi.bmiHeader.biCompression = BI_RGB;

        StretchDIBits(
            hdc,
            0, 0, w, h,
            0, 0, frame->width, frame->height,
            frame->data,
            &bmi,
            DIB_RGB_COLORS,
            SRCCOPY
        );
    } else {
        /* Dark Gradient Backdrop when waiting for frame */
        HBRUSH bg_brush = CreateSolidBrush(RGB(18, 20, 24));
        RECT r = {0, 0, w, h};
        FillRect(hdc, &r, bg_brush);
        DeleteObject(bg_brush);

        /* Grid lines for futuristic surveillance aesthetic */
        HPEN grid_pen = CreatePen(PS_SOLID, 1, RGB(30, 36, 46));
        HPEN old_pen = (HPEN)SelectObject(hdc, grid_pen);
        for (int x = 0; x < w; x += 40) {
            MoveToEx(hdc, x, 0, NULL);
            LineTo(hdc, x, h);
        }
        for (int y = 0; y < h; y += 40) {
            MoveToEx(hdc, 0, y, NULL);
            LineTo(hdc, w, y);
        }
        SelectObject(hdc, old_pen);
        DeleteObject(grid_pen);
    }

    SetBkMode(hdc, TRANSPARENT);

    /* 2. Virtual Line Crossing Render */
    int line_y = (int)((line_y_ratio > 0.0f) ? (h * line_y_ratio) : (h * 0.60f));
    HPEN line_pen = CreatePen(PS_DASH, 2, RGB(255, 180, 0));
    HPEN old_pen = (HPEN)SelectObject(hdc, line_pen);
    MoveToEx(hdc, 0, line_y, NULL);
    LineTo(hdc, w, line_y);
    SelectObject(hdc, old_pen);
    DeleteObject(line_pen);

    HFONT old_font = (HFONT)SelectObject(hdc, win->font_regular);
    SetTextColor(hdc, RGB(255, 180, 0));
    TextOutA(hdc, w - 240, line_y - 18, "--- VIRTUAL CROSSING LINE ---", 29);

    /* 3. Render Detected Faces & Bounding Boxes */
    if (faces && num_faces > 0) {
        for (int i = 0; i < num_faces; i++) {
            const FaceResult *face = &faces[i];
            int bx1 = (int)face->bbox.x1;
            int by1 = (int)face->bbox.y1;
            int bx2 = (int)face->bbox.x2;
            int by2 = (int)face->bbox.y2;

            if (bx2 <= bx1 || by2 <= by1) continue;

            bool is_match = face->has_embedding;
            COLORREF box_color = is_match ? RGB(0, 255, 128) : RGB(0, 215, 255);

            /* Draw Box Corner Accents */
            HPEN box_pen = CreatePen(PS_SOLID, 2, box_color);
            old_pen = (HPEN)SelectObject(hdc, box_pen);
            
            HBRUSH null_brush = (HBRUSH)GetStockObject(NULL_BRUSH);
            HBRUSH old_brush = (HBRUSH)SelectObject(hdc, null_brush);
            Rectangle(hdc, bx1, by1, bx2, by2);
            SelectObject(hdc, old_brush);

            SelectObject(hdc, old_pen);
            DeleteObject(box_pen);

            /* Draw Label Badge */
            char label[128];
            if (is_match) {
                snprintf(label, sizeof(label), "[CHECK-IN] MATCH (%.2f)", face->detection_score);
            } else {
                snprintf(label, sizeof(label), "UNKNOWN (%.2f)", face->detection_score);
            }

            RECT text_rect = {bx1, max(0, by1 - 22), bx1 + 180, by1};
            HBRUSH badge_brush = CreateSolidBrush(RGB(10, 10, 10));
            FillRect(hdc, &text_rect, badge_brush);
            DeleteObject(badge_brush);

            SelectObject(hdc, win->font_bold);
            SetTextColor(hdc, box_color);
            TextOutA(hdc, bx1 + 4, max(2, by1 - 18), label, (int)strlen(label));
        }
    }

    /* 4. Sleek HUD Performance Card (Top Left) */
    RECT hud_rect = {12, 12, 310, 100};
    HBRUSH hud_brush = CreateSolidBrush(RGB(15, 18, 24));
    FillRect(hdc, &hud_rect, hud_brush);
    DeleteObject(hud_brush);

    HPEN hud_border = CreatePen(PS_SOLID, 1, RGB(50, 60, 80));
    old_pen = (HPEN)SelectObject(hdc, hud_border);
    HBRUSH null_br = (HBRUSH)GetStockObject(NULL_BRUSH);
    HBRUSH old_br = (HBRUSH)SelectObject(hdc, null_br);
    Rectangle(hdc, hud_rect.left, hud_rect.top, hud_rect.right, hud_rect.bottom);
    SelectObject(hdc, old_br);
    SelectObject(hdc, old_pen);
    DeleteObject(hud_border);

    /* HUD Status Texts */
    SelectObject(hdc, win->font_bold);
    SetTextColor(hdc, RGB(0, 255, 128));
    TextOutA(hdc, 22, 20, "LIVE CAMERA STREAM [ACTIVE]", 27);

    SelectObject(hdc, win->font_regular);
    SetTextColor(hdc, RGB(220, 225, 230));

    char fps_txt[64];
    snprintf(fps_txt, sizeof(fps_txt), "Display FPS: %.1f FPS", display_fps);
    TextOutA(hdc, 22, 42, fps_txt, (int)strlen(fps_txt));

    char inf_txt[64];
    snprintf(inf_txt, sizeof(inf_txt), "Inference Time: %.1f ms", inference_ms);
    TextOutA(hdc, 22, 60, inf_txt, (int)strlen(inf_txt));

    char trk_txt[64];
    int track_count = tracker ? tracker->count : 0;
    snprintf(trk_txt, sizeof(trk_txt), "Active Tracks: %d | Faces: %d", track_count, num_faces);
    TextOutA(hdc, 22, 78, trk_txt, (int)strlen(trk_txt));

    /* 5. Attendance Event Toast Banner (Top Right or Bottom Center) */
    if (last_event_msg && last_event_msg[0] != '\0') {
        RECT toast_rect = {w / 2 - 220, 16, w / 2 + 220, 56};
        HBRUSH toast_bg = CreateSolidBrush(RGB(16, 120, 60));
        FillRect(hdc, &toast_rect, toast_bg);
        DeleteObject(toast_bg);

        SelectObject(hdc, win->font_bold);
        SetTextColor(hdc, RGB(255, 255, 255));
        TextOutA(hdc, toast_rect.left + 16, toast_rect.top + 10, last_event_msg, (int)strlen(last_event_msg));
    }

    SelectObject(hdc, old_font);

    /* Blit double-buffered frame onto the actual window */
    BitBlt(win->hdc, 0, 0, w, h, win->mem_dc, 0, 0, SRCCOPY);
}

#else
/* Non-Windows stub */
struct GuiWindow { int dummy; };
GuiWindow *gui_window_create(const char *title, int width, int height) { (void)title; (void)width; (void)height; return NULL; }
void gui_window_destroy(GuiWindow *win) { (void)win; }
bool gui_window_process_events(GuiWindow *win) { (void)win; return true; }
void gui_window_render(GuiWindow *win, const ImageBuffer *frame, const FaceResult *faces, int num_faces, const CentroidTracker *tracker, float line_y_ratio, float display_fps, float inference_ms, const char *last_event_msg) { (void)win; (void)frame; (void)faces; (void)num_faces; (void)tracker; (void)line_y_ratio; (void)display_fps; (void)inference_ms; (void)last_event_msg; }
#endif
