#include "attendance/gui_preview.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#ifdef _WIN32
#include <windows.h>
#endif

struct GuiWindow {
    char title[256];
    int width;
    int height;
#ifdef _WIN32
    HWND hwnd;
    HDC hdc_mem;
    HBITMAP hbm_mem;
    HBITMAP hbm_old;
    HFONT font_regular;
    HFONT font_bold;
    HFONT font_title;
    HFONT font_small;
#endif
    bool is_open;
};

#ifdef _WIN32
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_ERASEBKGND:
            return 1; /* Avoid flicker during resize/fullscreen */
        case WM_CLOSE:
            DestroyWindow(hwnd);
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE || wParam == 'Q' || wParam == 'q') {
                PostQuitMessage(0);
            }
            return 0;
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
}
#endif

static void format_person_name(const char *raw_name, char *out_name, size_t max_len) {
    if (!raw_name || raw_name[0] == '\0') {
        strncpy(out_name, "Unknown", max_len - 1);
        return;
    }
    size_t out_idx = 0;
    for (size_t i = 0; raw_name[i] != '\0' && out_idx + 2 < max_len; i++) {
        if (i > 0 && isupper((unsigned char)raw_name[i]) && !isupper((unsigned char)raw_name[i - 1])) {
            out_name[out_idx++] = ' ';
        }
        if (i == 0) {
            out_name[out_idx++] = (char)toupper((unsigned char)raw_name[i]);
        } else {
            out_name[out_idx++] = raw_name[i];
        }
    }
    out_name[out_idx] = '\0';
}

GuiWindow *gui_window_create(const char *title, int width, int height) {
    GuiWindow *win = (GuiWindow *)calloc(1, sizeof(GuiWindow));
    if (!win) return NULL;

    strncpy(win->title, title ? title : "Talangmas AI Surveillance", sizeof(win->title) - 1);
    win->width = (width > 0) ? width : 1280;
    win->height = (height > 0) ? height : 720;

#ifdef _WIN32
    HINSTANCE hInst = GetModuleHandle(NULL);

    WNDCLASSA wc;
    memset(&wc, 0, sizeof(wc));
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = "TalangmasPreviewWindowClass";
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    RegisterClassA(&wc);

    RECT wr = {0, 0, win->width, win->height};
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);

    win->hwnd = CreateWindowExA(
        0,
        wc.lpszClassName,
        win->title,
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        wr.right - wr.left, wr.bottom - wr.top,
        NULL, NULL, hInst, NULL
    );

    if (!win->hwnd) {
        free(win);
        return NULL;
    }

    HDC hdc_screen = GetDC(win->hwnd);
    win->hdc_mem = CreateCompatibleDC(hdc_screen);
    win->hbm_mem = CreateCompatibleBitmap(hdc_screen, win->width, win->height);
    win->hbm_old = (HBITMAP)SelectObject(win->hdc_mem, win->hbm_mem);
    ReleaseDC(win->hwnd, hdc_screen);

    win->font_regular = CreateFontA(
        15, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
        ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI"
    );

    win->font_bold = CreateFontA(
        16, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
        ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI"
    );

    win->font_small = CreateFontA(
        13, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI"
    );

    win->font_title = CreateFontA(
        20, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
        ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI"
    );

    win->is_open = true;
#endif

    return win;
}

void gui_window_destroy(GuiWindow *win) {
    if (!win) return;
#ifdef _WIN32
    if (win->hdc_mem) {
        SelectObject(win->hdc_mem, win->hbm_old);
        DeleteDC(win->hdc_mem);
    }
    if (win->hbm_mem) DeleteObject(win->hbm_mem);
    if (win->font_regular) DeleteObject(win->font_regular);
    if (win->font_bold) DeleteObject(win->font_bold);
    if (win->font_small) DeleteObject(win->font_small);
    if (win->font_title) DeleteObject(win->font_title);

    if (win->hwnd) {
        DestroyWindow(win->hwnd);
    }
#endif
    free(win);
}

bool gui_window_process_events(GuiWindow *win) {
    if (!win) return false;
#ifdef _WIN32
    MSG msg;
    while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            win->is_open = false;
            return false;
        }
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    return win->is_open;
#else
    return true;
#endif
}

void gui_window_render(
    GuiWindow *win,
    const ImageBuffer *frame,
    const FaceResult *faces,
    int num_faces,
    float display_fps,
    float inference_ms,
    const char *last_event_msg
) {
    (void)last_event_msg;
    if (!win) return;

#ifdef _WIN32
    if (!win->hwnd || !win->hdc_mem) return;

    /* Get dynamic current window client area */
    RECT client_rc;
    GetClientRect(win->hwnd, &client_rc);
    int w = client_rc.right - client_rc.left;
    int h = client_rc.bottom - client_rc.top;
    if (w <= 0 || h <= 0) return;

    /* Resize offscreen buffer if window size changed (fullscreen / maximize) */
    if (w != win->width || h != win->height || !win->hbm_mem) {
        win->width = w;
        win->height = h;
        if (win->hbm_mem) {
            SelectObject(win->hdc_mem, win->hbm_old);
            DeleteObject(win->hbm_mem);
        }
        HDC hdc_screen = GetDC(win->hwnd);
        win->hbm_mem = CreateCompatibleBitmap(hdc_screen, w, h);
        win->hbm_old = (HBITMAP)SelectObject(win->hdc_mem, win->hbm_mem);
        ReleaseDC(win->hwnd, hdc_screen);
    }

    HDC hdc = win->hdc_mem;

    /* Coordinate scaling factors from source frame resolution to current display window */
    float scale_x = 1.0f;
    float scale_y = 1.0f;

    /* 1. Paint Raw Camera Video Buffer Scaled to Full Window */
    if (frame && frame->data && frame->width > 0 && frame->height > 0) {
        scale_x = (float)w / (float)frame->width;
        scale_y = (float)h / (float)frame->height;

        BITMAPINFO bmi;
        memset(&bmi, 0, sizeof(bmi));
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = frame->width;
        bmi.bmiHeader.biHeight = -frame->height; /* Top-down BGR */
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 24;
        bmi.bmiHeader.biCompression = BI_RGB;

        SetStretchBltMode(hdc, HALFTONE);
        SetBrushOrgEx(hdc, 0, 0, NULL);
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
        HBRUSH bg_brush = CreateSolidBrush(RGB(18, 20, 24));
        RECT r = {0, 0, w, h};
        FillRect(hdc, &r, bg_brush);
        DeleteObject(bg_brush);
    }

    SetBkMode(hdc, TRANSPARENT);
    HFONT old_font = (HFONT)SelectObject(hdc, win->font_regular);

    /* 2. Render Detected Faces & Person Names (Scaled to Full Window) */
    if (faces && num_faces > 0) {
        for (int i = 0; i < num_faces; i++) {
            const FaceResult *face = &faces[i];
            int bx1 = (int)(face->bbox.x1 * scale_x);
            int by1 = (int)(face->bbox.y1 * scale_y);
            int bx2 = (int)(face->bbox.x2 * scale_x);
            int by2 = (int)(face->bbox.y2 * scale_y);

            if (bx2 <= bx1 || by2 <= by1) continue;

            bool is_match = face->is_recognized && (face->matched_name[0] != '\0');
            COLORREF box_color = is_match ? RGB(0, 255, 128) : RGB(0, 215, 255);
            COLORREF badge_bg = is_match ? RGB(10, 30, 20) : RGB(15, 22, 32);

            /* 2a. HUD Corner Brackets on Bounding Box */
            int corner_len = (bx2 - bx1) / 4;
            if (corner_len < 10) corner_len = 10;
            if (corner_len > 30) corner_len = 30;

            HPEN box_pen = CreatePen(PS_SOLID, 2, box_color);
            HPEN old_pen = (HPEN)SelectObject(hdc, box_pen);

            /* Top-Left */
            MoveToEx(hdc, bx1, by1 + corner_len, NULL); LineTo(hdc, bx1, by1); LineTo(hdc, bx1 + corner_len, by1);
            /* Top-Right */
            MoveToEx(hdc, bx2 - corner_len, by1, NULL); LineTo(hdc, bx2, by1); LineTo(hdc, bx2, by1 + corner_len);
            /* Bottom-Left */
            MoveToEx(hdc, bx1, by2 - corner_len, NULL); LineTo(hdc, bx1, by2); LineTo(hdc, bx1 + corner_len, by2);
            /* Bottom-Right */
            MoveToEx(hdc, bx2 - corner_len, by2, NULL); LineTo(hdc, bx2, by2); LineTo(hdc, bx2, by2 - corner_len);

            /* Thin outline frame */
            HPEN thin_pen = CreatePen(PS_DOT, 1, box_color);
            SelectObject(hdc, thin_pen);
            HBRUSH null_brush = (HBRUSH)GetStockObject(NULL_BRUSH);
            HBRUSH old_brush = (HBRUSH)SelectObject(hdc, null_brush);
            Rectangle(hdc, bx1, by1, bx2, by2);
            SelectObject(hdc, old_brush);
            SelectObject(hdc, old_pen);
            DeleteObject(box_pen);
            DeleteObject(thin_pen);

            /* 2b. Facial Landmarks (5 Points Scaled) */
            HPEN lm_pen = CreatePen(PS_SOLID, 1, RGB(255, 220, 60));
            old_pen = (HPEN)SelectObject(hdc, lm_pen);
            for (int k = 0; k < 5; k++) {
                int lx = (int)(face->landmarks.x[k] * scale_x);
                int ly = (int)(face->landmarks.y[k] * scale_y);
                if (lx >= bx1 && lx <= bx2 && ly >= by1 && ly <= by2) {
                    MoveToEx(hdc, lx - 3, ly, NULL); LineTo(hdc, lx + 4, ly);
                    MoveToEx(hdc, lx, ly - 3, NULL); LineTo(hdc, lx, ly + 4);
                }
            }
            SelectObject(hdc, old_pen);
            DeleteObject(lm_pen);

            /* 2c. Clean Person Name Label Badge */
            char disp_name[128];
            if (is_match) {
                format_person_name(face->matched_name, disp_name, sizeof(disp_name));
            } else {
                strncpy(disp_name, "Unknown", sizeof(disp_name) - 1);
            }

            /* Calculate text dimensions */
            SelectObject(hdc, win->font_bold);
            SIZE text_size;
            GetTextExtentPoint32A(hdc, disp_name, (int)strlen(disp_name), &text_size);

            int badge_w = text_size.cx + 20;
            if (badge_w < (bx2 - bx1)) badge_w = (bx2 - bx1);
            int badge_h = 26;
            int badge_y = by1 - badge_h - 4;
            if (badge_y < 4) badge_y = 4;

            RECT badge_rect = {bx1, badge_y, bx1 + badge_w, badge_y + badge_h};
            HBRUSH badge_b = CreateSolidBrush(badge_bg);
            FillRect(hdc, &badge_rect, badge_b);
            DeleteObject(badge_b);

            HPEN badge_border = CreatePen(PS_SOLID, 1, box_color);
            old_pen = (HPEN)SelectObject(hdc, badge_border);
            old_brush = (HBRUSH)SelectObject(hdc, null_brush);
            Rectangle(hdc, badge_rect.left, badge_rect.top, badge_rect.right, badge_rect.bottom);
            SelectObject(hdc, old_brush);
            SelectObject(hdc, old_pen);
            DeleteObject(badge_border);

            /* Render Name Text */
            SetTextColor(hdc, is_match ? RGB(0, 255, 128) : RGB(0, 215, 255));
            TextOutA(hdc, bx1 + 10, badge_y + 4, disp_name, (int)strlen(disp_name));
        }
    }

    /* 3. Sleek Minimal HUD Card (ONLY FPS and Inference Time) */
    RECT hud_rect = {16, 16, 195, 72};
    HBRUSH hud_brush = CreateSolidBrush(RGB(15, 18, 24));
    FillRect(hdc, &hud_rect, hud_brush);
    DeleteObject(hud_brush);

    HPEN hud_border = CreatePen(PS_SOLID, 1, RGB(45, 55, 75));
    HPEN old_pen = (HPEN)SelectObject(hdc, hud_border);
    HBRUSH null_br = (HBRUSH)GetStockObject(NULL_BRUSH);
    HBRUSH old_br = (HBRUSH)SelectObject(hdc, null_br);
    Rectangle(hdc, hud_rect.left, hud_rect.top, hud_rect.right, hud_rect.bottom);
    SelectObject(hdc, old_br);
    SelectObject(hdc, old_pen);
    DeleteObject(hud_border);

    /* Display FPS */
    SelectObject(hdc, win->font_bold);
    SetTextColor(hdc, RGB(0, 255, 128));
    char fps_txt[48];
    snprintf(fps_txt, sizeof(fps_txt), "FPS: %.1f", display_fps);
    TextOutA(hdc, 26, 23, fps_txt, (int)strlen(fps_txt));

    /* Display Inference Time */
    SetTextColor(hdc, RGB(0, 215, 255));
    char inf_txt[48];
    snprintf(inf_txt, sizeof(inf_txt), "Inference: %.1f ms", inference_ms);
    TextOutA(hdc, 26, 45, inf_txt, (int)strlen(inf_txt));

    SelectObject(hdc, old_font);

    /* Flush double buffer to screen */
    HDC hdc_screen = GetDC(win->hwnd);
    BitBlt(hdc_screen, 0, 0, w, h, win->hdc_mem, 0, 0, SRCCOPY);
    ReleaseDC(win->hwnd, hdc_screen);

#endif
}
