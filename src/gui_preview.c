#include "attendance/gui_preview.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#ifdef _WIN32
#include <windows.h>

typedef struct {
    char name[128];
    HBITMAP hbm_crop;
} FaceImageCache;

#define MAX_FACE_CACHE 128
static FaceImageCache face_cache[MAX_FACE_CACHE];
static int num_face_cache = 0;

static FaceImageCache* get_enrolled_face(const char *name) {
    if (!name || name[0] == '\0') return NULL;
    for (int i = 0; i < num_face_cache; i++) {
        if (strcmp(face_cache[i].name, name) == 0) {
            return face_cache[i].hbm_crop ? &face_cache[i] : NULL;
        }
    }
    
    if (num_face_cache >= MAX_FACE_CACHE) return NULL;
    
    char search_path[512];
    snprintf(search_path, sizeof(search_path), "data\\enroll\\%s\\*.jpg", name);
    
    WIN32_FIND_DATAA find_data;
    HANDLE hFind = FindFirstFileA(search_path, &find_data);
    if (hFind == INVALID_HANDLE_VALUE) {
        snprintf(search_path, sizeof(search_path), "data\\enroll\\%s\\*.png", name);
        hFind = FindFirstFileA(search_path, &find_data);
        if (hFind == INVALID_HANDLE_VALUE) {
            strcpy(face_cache[num_face_cache].name, name);
            face_cache[num_face_cache].hbm_crop = NULL;
            num_face_cache++;
            return NULL;
        }
    }
    
    char img_path[512];
    snprintf(img_path, sizeof(img_path), "data\\enroll\\%s\\%s", name, find_data.cFileName);
    FindClose(hFind);
    
    int w, h, c;
    uint8_t *data = stbi_load(img_path, &w, &h, &c, 4); // Force RGBA
    HBITMAP hbm = NULL;

    if (data) {
        // Swap R and B to make it BGRA for GDI
        for (int i = 0; i < w * h * 4; i += 4) {
            uint8_t temp = data[i];
            data[i] = data[i + 2];
            data[i + 2] = temp;
        }

        /* Pre-scale to 52x52 using GDI and store as HBITMAP */
        HDC hdc_screen = GetDC(NULL);
        hbm = CreateCompatibleBitmap(hdc_screen, 52, 52);
        HDC hdc_temp = CreateCompatibleDC(hdc_screen);
        HBITMAP old_bm = (HBITMAP)SelectObject(hdc_temp, hbm);

        BITMAPINFO cbmi;
        memset(&cbmi, 0, sizeof(cbmi));
        cbmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        cbmi.bmiHeader.biWidth = w;
        cbmi.bmiHeader.biHeight = -h; /* Top-down */
        cbmi.bmiHeader.biPlanes = 1;
        cbmi.bmiHeader.biBitCount = 32;
        cbmi.bmiHeader.biCompression = BI_RGB;

        SetStretchBltMode(hdc_temp, HALFTONE);
        SetBrushOrgEx(hdc_temp, 0, 0, NULL);
        StretchDIBits(
            hdc_temp,
            0, 0, 52, 52,
            0, 0, w, h,
            data,
            &cbmi,
            DIB_RGB_COLORS,
            SRCCOPY
        );

        SelectObject(hdc_temp, old_bm);
        DeleteDC(hdc_temp);
        ReleaseDC(NULL, hdc_screen);

        stbi_image_free(data); // Free the huge raw data!
    }
    
    FaceImageCache *cache = &face_cache[num_face_cache++];
    strcpy(cache->name, name);
    cache->hbm_crop = hbm;
    
    return cache->hbm_crop ? cache : NULL;
}

static HBITMAP create_snapshot_hbitmap(const ImageBuffer *frame, const FaceBBox *bbox, int target_w, int target_h) {
    if (!frame || !frame->data || frame->width <= 0 || frame->height <= 0 || !bbox) return NULL;

    float bw = bbox->x2 - bbox->x1;
    float bh = bbox->y2 - bbox->y1;
    if (bw <= 4.0f || bh <= 4.0f) return NULL;

    /* Add 15% padding around face for clean head crop */
    float pad_x = bw * 0.15f;
    float pad_y = bh * 0.15f;

    int fx1 = (int)fmaxf(0.0f, bbox->x1 - pad_x);
    int fy1 = (int)fmaxf(0.0f, bbox->y1 - pad_y);
    int fx2 = (int)fminf((float)frame->width, bbox->x2 + pad_x);
    int fy2 = (int)fminf((float)frame->height, bbox->y2 + pad_y);

    int crop_w = fx2 - fx1;
    int crop_h = fy2 - fy1;
    if (crop_w <= 4 || crop_h <= 4) return NULL;

    /* Manually crop & resample BGR pixels directly from top-down frame buffer */
    uint8_t *crop_bgra = (uint8_t *)calloc(1, target_w * target_h * 4);
    if (!crop_bgra) return NULL;

    int src_stride = frame->stride;
    int ch = frame->channels;

    for (int dy = 0; dy < target_h; dy++) {
        int sy = fy1 + (dy * crop_h) / target_h;
        if (sy >= frame->height) sy = frame->height - 1;
        const uint8_t *src_row = frame->data + sy * src_stride;

        for (int dx = 0; dx < target_w; dx++) {
            int sx = fx1 + (dx * crop_w) / target_w;
            if (sx >= frame->width) sx = frame->width - 1;

            const uint8_t *pixel = src_row + sx * ch;
            int dst_idx = (dy * target_w + dx) * 4;

            /* BGR to BGRA for GDI */
            crop_bgra[dst_idx + 0] = pixel[0]; // B
            crop_bgra[dst_idx + 1] = pixel[1]; // G
            crop_bgra[dst_idx + 2] = pixel[2]; // R
            crop_bgra[dst_idx + 3] = 255;      // A
        }
    }

    BITMAPINFO cbmi;
    memset(&cbmi, 0, sizeof(cbmi));
    cbmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    cbmi.bmiHeader.biWidth = target_w;
    cbmi.bmiHeader.biHeight = -target_h; /* Top-down BGRA */
    cbmi.bmiHeader.biPlanes = 1;
    cbmi.bmiHeader.biBitCount = 32;
    cbmi.bmiHeader.biCompression = BI_RGB;

    HDC hdc_screen = GetDC(NULL);
    HBITMAP hbm = CreateCompatibleBitmap(hdc_screen, target_w, target_h);
    HDC hdc_temp = CreateCompatibleDC(hdc_screen);
    HBITMAP old_bm = (HBITMAP)SelectObject(hdc_temp, hbm);

    SetStretchBltMode(hdc_temp, HALFTONE);
    SetBrushOrgEx(hdc_temp, 0, 0, NULL);
    StretchDIBits(
        hdc_temp,
        0, 0, target_w, target_h,
        0, 0, target_w, target_h,
        crop_bgra,
        &cbmi,
        DIB_RGB_COLORS,
        SRCCOPY
    );

    SelectObject(hdc_temp, old_bm);
    DeleteDC(hdc_temp);
    ReleaseDC(NULL, hdc_screen);
    free(crop_bgra);

    return hbm;
}
#endif

typedef struct {
    char matched_name[128];
    float max_score;
    ULONGLONG last_seen_tick;
    HBITMAP hbm_snapshot;
} PanelPerson;

#define MAX_PANEL_PERSONS 32
static PanelPerson panel_persons[MAX_PANEL_PERSONS];
static int num_panel_persons = 0;

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

    /* Define Side Panel Width */
    /* Define Side Panel Width */
    int panel_w = 340;
    int cam_w = (w > panel_w + 320) ? (w - panel_w) : w;
    bool has_panel = (cam_w != w);

    /* Coordinate scaling factors from source frame resolution to current display window */
    float scale_x = 1.0f;
    float scale_y = 1.0f;

    BITMAPINFO bmi;
    if (frame && frame->data && frame->width > 0 && frame->height > 0) {
        memset(&bmi, 0, sizeof(bmi));
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = frame->width;
        bmi.bmiHeader.biHeight = -frame->height; /* Top-down BGR */
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 24;
        bmi.bmiHeader.biCompression = BI_RGB;
    }

    /* 1. Paint Raw Camera Video Buffer Scaled to Camera Window */
    if (frame && frame->data && frame->width > 0 && frame->height > 0) {
        scale_x = (float)cam_w / (float)frame->width;
        scale_y = (float)h / (float)frame->height;

        SetStretchBltMode(hdc, HALFTONE);
        SetBrushOrgEx(hdc, 0, 0, NULL);
        StretchDIBits(
            hdc,
            0, 0, cam_w, h,
            0, 0, frame->width, frame->height,
            frame->data,
            &bmi,
            DIB_RGB_COLORS,
            SRCCOPY
        );
    } else {
        HBRUSH bg_brush = CreateSolidBrush(RGB(18, 20, 24));
        RECT r = {0, 0, cam_w, h};
        FillRect(hdc, &r, bg_brush);
        DeleteObject(bg_brush);
    }

    /* 1.5. Paint Side Panel Background */
    if (has_panel) {
        HBRUSH panel_bg = CreateSolidBrush(RGB(22, 26, 33));
        RECT r_panel = {cam_w, 0, w, h};
        FillRect(hdc, &r_panel, panel_bg);
        DeleteObject(panel_bg);

        /* Panel Title */
        SetBkMode(hdc, TRANSPARENT);
        SelectObject(hdc, win->font_title);
        SetTextColor(hdc, RGB(255, 255, 255));
        TextOutA(hdc, cam_w + 16, 16, "Detected Persons", 16);
    }

    SetBkMode(hdc, TRANSPARENT);
    HFONT old_font = (HFONT)SelectObject(hdc, win->font_regular);

    /* 2. Render Detected Faces & Person Names (Scaled to Full Window) */
    int panel_cursor_y = 50; /* Start Y position for items in the side panel */

    if (faces && num_faces > 0) {
        for (int i = 0; i < num_faces; i++) {
            const FaceResult *face = &faces[i];
            int bx1 = (int)(face->bbox.x1 * scale_x);
            int by1 = (int)(face->bbox.y1 * scale_y);
            int bx2 = (int)(face->bbox.x2 * scale_x);
            int by2 = (int)(face->bbox.y2 * scale_y);

            if (bx2 <= bx1 || by2 <= by1) continue;

            bool is_match = face->is_recognized && (face->matched_name[0] != '\0') && (face->match_score >= 0.45f);
            COLORREF box_color = is_match ? RGB(0, 255, 128) : RGB(0, 215, 255);

            if (is_match) {
                ULONGLONG current_tick = GetTickCount64();
                int found_idx = -1;
                for (int p = 0; p < num_panel_persons; p++) {
                    if (strcmp(panel_persons[p].matched_name, face->matched_name) == 0) {
                        panel_persons[p].last_seen_tick = current_tick;
                        if (face->match_score > panel_persons[p].max_score) {
                            panel_persons[p].max_score = face->match_score;
                        }
                        found_idx = p;
                        break;
                    }
                }

                if (found_idx >= 0) {
                    /* Move recently recognized person to top (index 0) */
                    if (found_idx > 0) {
                        PanelPerson temp = panel_persons[found_idx];
                        for (int k = found_idx; k > 0; k--) {
                            panel_persons[k] = panel_persons[k - 1];
                        }
                        panel_persons[0] = temp;
                    }
                    /* Ensure still snapshot photo exists (captured once) */
                    if (!panel_persons[0].hbm_snapshot) {
                        panel_persons[0].hbm_snapshot = create_snapshot_hbitmap(frame, &face->bbox, 52, 52);
                    }
                } else {
                    /* Insert new person at index 0 (top of list) and capture STILL snapshot ONCE */
                    int limit = (num_panel_persons < MAX_PANEL_PERSONS) ? num_panel_persons : (MAX_PANEL_PERSONS - 1);
                    if (num_panel_persons == MAX_PANEL_PERSONS && panel_persons[limit].hbm_snapshot) {
                        DeleteObject(panel_persons[limit].hbm_snapshot);
                    }
                    for (int k = limit; k > 0; k--) {
                        panel_persons[k] = panel_persons[k - 1];
                    }
                    strcpy(panel_persons[0].matched_name, face->matched_name);
                    panel_persons[0].max_score = face->match_score;
                    panel_persons[0].last_seen_tick = current_tick;
                    panel_persons[0].hbm_snapshot = create_snapshot_hbitmap(frame, &face->bbox, 52, 52);
                    if (num_panel_persons < MAX_PANEL_PERSONS) {
                        num_panel_persons++;
                    }
                }
            }

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

        }
    }

    /* 2.5 Draw on Side Panel using panel_persons state (10s freeze) */
    if (has_panel) {
        ULONGLONG current_tick = GetTickCount64();
        for (int p = 0; p < num_panel_persons; p++) {
            if (current_tick - panel_persons[p].last_seen_tick > 10000) {
                if (panel_persons[p].hbm_snapshot) {
                    DeleteObject(panel_persons[p].hbm_snapshot);
                    panel_persons[p].hbm_snapshot = NULL;
                }
                // Remove expired person by shifting array left
                for (int j = p; j < num_panel_persons - 1; j++) {
                    panel_persons[j] = panel_persons[j+1];
                }
                num_panel_persons--;
                p--; // re-check this index
                continue;
            }

            if (panel_cursor_y + 65 >= h) break; // Out of space

            int px1 = cam_w + 14;
            int px2 = cam_w + 72;
            int tx = cam_w + 132;
            int py = panel_cursor_y;
            int pw = 52;
            int ph = 52;
            COLORREF badge_bg = RGB(10, 30, 20);
            COLORREF box_color = RGB(0, 255, 128);
            COLORREF snap_border = RGB(0, 215, 255);

            char disp_name[128];
            format_person_name(panel_persons[p].matched_name, disp_name, sizeof(disp_name));

            FaceImageCache *cache = get_enrolled_face(panel_persons[p].matched_name);

            /* 1. Enrolled Profile Photo (Green Border) */
            RECT crop1_bg = {px1 - 1, py - 1, px1 + pw + 1, py + ph + 1};
            HBRUSH b1 = CreateSolidBrush(badge_bg);
            FillRect(hdc, &crop1_bg, b1);
            DeleteObject(b1);

            if (cache && cache->hbm_crop) {
                HDC hdc_face = CreateCompatibleDC(hdc);
                HBITMAP old_bm = (HBITMAP)SelectObject(hdc_face, cache->hbm_crop);
                BitBlt(hdc, px1, py, pw, ph, hdc_face, 0, 0, SRCCOPY);
                SelectObject(hdc_face, old_bm);
                DeleteDC(hdc_face);
            } else {
                SelectObject(hdc, win->font_small);
                SetTextColor(hdc, RGB(100, 100, 100));
                TextOutA(hdc, px1 + 6, py + 18, "No Pic", 6);
            }

            HPEN pen1 = CreatePen(PS_SOLID, 1, box_color);
            HPEN old_pen1 = (HPEN)SelectObject(hdc, pen1);
            HBRUSH null_b1 = (HBRUSH)GetStockObject(NULL_BRUSH);
            HBRUSH old_br1 = (HBRUSH)SelectObject(hdc, null_b1);
            Rectangle(hdc, px1, py, px1 + pw, py + ph);
            SelectObject(hdc, old_br1);
            SelectObject(hdc, old_pen1);
            DeleteObject(pen1);

            /* 2. Live Camera Snapshot Photo (Cyan Border) */
            RECT crop2_bg = {px2 - 1, py - 1, px2 + pw + 1, py + ph + 1};
            HBRUSH b2 = CreateSolidBrush(RGB(15, 22, 32));
            FillRect(hdc, &crop2_bg, b2);
            DeleteObject(b2);

            if (panel_persons[p].hbm_snapshot) {
                HDC hdc_snap = CreateCompatibleDC(hdc);
                HBITMAP old_bm = (HBITMAP)SelectObject(hdc_snap, panel_persons[p].hbm_snapshot);
                BitBlt(hdc, px2, py, pw, ph, hdc_snap, 0, 0, SRCCOPY);
                SelectObject(hdc_snap, old_bm);
                DeleteDC(hdc_snap);
            } else {
                SelectObject(hdc, win->font_small);
                SetTextColor(hdc, RGB(100, 100, 100));
                TextOutA(hdc, px2 + 6, py + 18, "No Snap", 7);
            }

            HPEN pen2 = CreatePen(PS_SOLID, 1, snap_border);
            HPEN old_pen2 = (HPEN)SelectObject(hdc, pen2);
            HBRUSH null_b2 = (HBRUSH)GetStockObject(NULL_BRUSH);
            HBRUSH old_br2 = (HBRUSH)SelectObject(hdc, null_b2);
            Rectangle(hdc, px2, py, px2 + pw, py + ph);
            SelectObject(hdc, old_br2);
            SelectObject(hdc, old_pen2);
            DeleteObject(pen2);

            /* 3. Render Name & Score */
            SelectObject(hdc, win->font_bold);
            SetTextColor(hdc, RGB(0, 255, 128));
            TextOutA(hdc, tx, py + 4, disp_name, (int)strlen(disp_name));
            
            SelectObject(hdc, win->font_small);
            SetTextColor(hdc, RGB(150, 150, 150));
            char acc_str[32];
            snprintf(acc_str, sizeof(acc_str), "Score: %.2f", panel_persons[p].max_score);
            TextOutA(hdc, tx, py + 26, acc_str, (int)strlen(acc_str));

            panel_cursor_y += ph + 14;
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
