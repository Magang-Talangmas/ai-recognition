#include "attendance/gui_preview.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#define CANONICAL_CANVAS_W 1280
#define CANONICAL_CANVAS_H 720
#define CANONICAL_PANEL_W  280

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
    
    const char *prefixes[] = {
        "data\\enroll\\%s\\",
        "data/enroll/%s/",
        "..\\data\\enroll\\%s\\",
        NULL
    };
    const char *exts[] = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", NULL};
    
    char found_dir[512] = {0};
    WIN32_FIND_DATAA find_data;
    HANDLE hFind = INVALID_HANDLE_VALUE;

    for (int i = 0; prefixes[i] != NULL && hFind == INVALID_HANDLE_VALUE; i++) {
        char dir_buf[512];
        snprintf(dir_buf, sizeof(dir_buf), prefixes[i], name);
        for (int j = 0; exts[j] != NULL; j++) {
            char search_path[512];
            snprintf(search_path, sizeof(search_path), "%s%s", dir_buf, exts[j]);
            hFind = FindFirstFileA(search_path, &find_data);
            if (hFind != INVALID_HANDLE_VALUE) {
                strncpy(found_dir, dir_buf, sizeof(found_dir) - 1);
                break;
            }
        }
    }

    if (hFind == INVALID_HANDLE_VALUE) {
        strcpy(face_cache[num_face_cache].name, name);
        face_cache[num_face_cache].hbm_crop = NULL;
        num_face_cache++;
        return NULL;
    }
    
    char img_path[512];
    snprintf(img_path, sizeof(img_path), "%s%s", found_dir, find_data.cFileName);
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

        /* Pre-scale to 80x80 using GDI and store as HBITMAP */
        HDC hdc_screen = GetDC(NULL);
        hbm = CreateCompatibleBitmap(hdc_screen, 80, 80);
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
            0, 0, 80, 80,
            0, 0, w, h,
            data,
            &cbmi,
            DIB_RGB_COLORS,
            SRCCOPY
        );

        SelectObject(hdc_temp, old_bm);
        DeleteDC(hdc_temp);
        ReleaseDC(NULL, hdc_screen);

        stbi_image_free(data);
    }
    
    FaceImageCache *cache = &face_cache[num_face_cache++];
    strcpy(cache->name, name);
    cache->hbm_crop = hbm;
    
    return cache->hbm_crop ? cache : NULL;
}
#endif

typedef struct {
    char matched_name[128];
    float max_score;
    ULONGLONG last_seen_tick;
} PanelPerson;

#define MAX_PANEL_PERSONS 32
static PanelPerson panel_persons[MAX_PANEL_PERSONS];
static int num_panel_persons = 0;

struct GuiWindow {
    char title[256];
    int width;
    int height;
#ifdef _WIN32
    HWND hwnd;
    HDC hdc_mem;
    HBITMAP hbm_mem;
    HBITMAP hbm_old;
    HANDLE h_shm_map;
    uint8_t *shm_ptr;
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
        case WM_GETMINMAXINFO: {
            LPMINMAXINFO lpMMI = (LPMINMAXINFO)lParam;
            lpMMI->ptMinTrackSize.x = 640;
            lpMMI->ptMinTrackSize.y = 360;
            return 0;
        }
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

    strncpy(win->title, title ? title : "Talangmas AI Attendance - Live View", sizeof(win->title) - 1);
    win->width = (width > 0) ? width : CANONICAL_CANVAS_W;
    win->height = (height > 0) ? height : CANONICAL_CANVAS_H;

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

    /* Fixed canonical canvas size (1280x720) for consistent rendering & web stream */
    HDC hdc_screen = GetDC(win->hwnd);
    win->hdc_mem = CreateCompatibleDC(hdc_screen);
    win->hbm_mem = CreateCompatibleBitmap(hdc_screen, CANONICAL_CANVAS_W, CANONICAL_CANVAS_H);
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

    /* Setup Shared Memory for zero-latency Web/Mobile Live Streaming */
    win->h_shm_map = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        16 * 1024 * 1024,
        "Local\\TMAS_PREVIEW_SHM"
    );
    if (win->h_shm_map) {
        win->shm_ptr = (uint8_t *)MapViewOfFile(
            win->h_shm_map,
            FILE_MAP_ALL_ACCESS,
            0,
            0,
            16 * 1024 * 1024
        );
    }

    win->is_open = true;
#endif

    return win;
}

void gui_window_destroy(GuiWindow *win) {
    if (!win) return;
#ifdef _WIN32
    if (win->shm_ptr) {
        UnmapViewOfFile(win->shm_ptr);
        win->shm_ptr = NULL;
    }
    if (win->h_shm_map) {
        CloseHandle(win->h_shm_map);
        win->h_shm_map = NULL;
    }

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

    HDC hdc = win->hdc_mem;
    int can_w = CANONICAL_CANVAS_W;
    int can_h = CANONICAL_CANVAS_H;
    int panel_w = CANONICAL_PANEL_W;
    int cam_viewport_w = can_w; /* 1280px viewport (FULLSCREEN) */
    int cam_viewport_h = can_h;           /* 720px viewport */

    /* 1. Paint Camera Viewport Background */
    RECT cam_vp_rect = {0, 0, cam_viewport_w, cam_viewport_h};
    HBRUSH cam_bg_brush = CreateSolidBrush(RGB(15, 18, 24));
    FillRect(hdc, &cam_vp_rect, cam_bg_brush);
    DeleteObject(cam_bg_brush);

    /* 1.1 Calculate Aspect-Ratio-Preserving Coordinates for Camera Frame */
    int cam_draw_w = cam_viewport_w;
    int cam_draw_h = cam_viewport_h;
    int cam_offset_x = 0;
    int cam_offset_y = 0;
    float scale_x = 1.0f;
    float scale_y = 1.0f;

    if (frame && frame->data && frame->width > 0 && frame->height > 0) {
        float aspect_src = (float)frame->width / (float)frame->height;
        float aspect_dst = (float)cam_viewport_w / (float)cam_viewport_h;

        if (aspect_src > aspect_dst) {
            cam_draw_w = cam_viewport_w;
            cam_draw_h = (int)((float)cam_viewport_w / aspect_src);
            cam_offset_x = 0;
            cam_offset_y = (cam_viewport_h - cam_draw_h) / 2;
        } else {
            cam_draw_h = cam_viewport_h;
            cam_draw_w = (int)((float)cam_viewport_h * aspect_src);
            cam_offset_x = (cam_viewport_w - cam_draw_w) / 2;
            cam_offset_y = 0;
        }

        scale_x = (float)cam_draw_w / (float)frame->width;
        scale_y = (float)cam_draw_h / (float)frame->height;

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
            cam_offset_x, cam_offset_y, cam_draw_w, cam_draw_h,
            0, 0, frame->width, frame->height,
            frame->data,
            &bmi,
            DIB_RGB_COLORS,
            SRCCOPY
        );
    }

    SetBkMode(hdc, TRANSPARENT);
    HFONT old_font = (HFONT)SelectObject(hdc, win->font_regular);

    /* 2. Render Detected Faces & Person Names (Scaled to Camera Viewport) */
    int panel_cursor_y = 56;

    if (faces && num_faces > 0) {
        for (int i = 0; i < num_faces; i++) {
            const FaceResult *face = &faces[i];
            int bx1 = cam_offset_x + (int)(face->bbox.x1 * scale_x);
            int by1 = cam_offset_y + (int)(face->bbox.y1 * scale_y);
            int bx2 = cam_offset_x + (int)(face->bbox.x2 * scale_x);
            int by2 = cam_offset_y + (int)(face->bbox.y2 * scale_y);

            if (bx2 <= bx1 || by2 <= by1) continue;

            bool is_match = face->is_recognized && (face->matched_name[0] != '\0');
            COLORREF box_color = is_match ? RGB(0, 255, 128) : RGB(0, 215, 255);

            if (is_match) {
                ULONGLONG current_tick = GetTickCount64();
                bool found = false;
                for (int p = 0; p < num_panel_persons; p++) {
                    if (strcmp(panel_persons[p].matched_name, face->matched_name) == 0) {
                        panel_persons[p].last_seen_tick = current_tick;
                        if (face->match_score > panel_persons[p].max_score) {
                            panel_persons[p].max_score = face->match_score;
                        }
                        found = true;
                        break;
                    }
                }
                if (!found && num_panel_persons < MAX_PANEL_PERSONS) {
                    strcpy(panel_persons[num_panel_persons].matched_name, face->matched_name);
                    panel_persons[num_panel_persons].max_score = face->match_score;
                    panel_persons[num_panel_persons].last_seen_tick = current_tick;
                    num_panel_persons++;
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
        }
    }

    SelectObject(hdc, old_font);

    /* 4. Export Fixed Canonical 1280x720 Frame to Shared Memory for Web & Mobile */
    if (win->shm_ptr) {
        BITMAPINFO shm_bmi;
        memset(&shm_bmi, 0, sizeof(shm_bmi));
        shm_bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        shm_bmi.bmiHeader.biWidth = CANONICAL_CANVAS_W;
        shm_bmi.bmiHeader.biHeight = -CANONICAL_CANVAS_H; /* Top-down BGR */
        shm_bmi.bmiHeader.biPlanes = 1;
        shm_bmi.bmiHeader.biBitCount = 24;
        shm_bmi.bmiHeader.biCompression = BI_RGB;

        uint8_t *pixels = win->shm_ptr + 24; /* 24 bytes header offset */
        GetDIBits(win->hdc_mem, win->hbm_mem, 0, CANONICAL_CANVAS_H, pixels, &shm_bmi, DIB_RGB_COLORS);

        /* Write 24-byte header: uint32 magic (TMAS), uint32 width, uint32 height, uint32 channels, uint64 seq */
        uint32_t *hdr32 = (uint32_t *)win->shm_ptr;
        uint64_t *hdr64 = (uint64_t *)(win->shm_ptr + 16);
        hdr32[1] = (uint32_t)CANONICAL_CANVAS_W;
        hdr32[2] = (uint32_t)CANONICAL_CANVAS_H;
        hdr32[3] = 3;
        (*hdr64)++;
        hdr32[0] = 0x53414D54; /* 'TMAS' magic */
    }

    /* 5. Letterbox / Pillarbox Scale to Desktop Window Client Area */
    RECT client_rc;
    GetClientRect(win->hwnd, &client_rc);
    int win_w = client_rc.right - client_rc.left;
    int win_h = client_rc.bottom - client_rc.top;

    if (win_w > 0 && win_h > 0) {
        float scale_fit = ((float)win_w / (float)CANONICAL_CANVAS_W < (float)win_h / (float)CANONICAL_CANVAS_H) ?
                          ((float)win_w / (float)CANONICAL_CANVAS_W) :
                          ((float)win_h / (float)CANONICAL_CANVAS_H);

        int dest_w = (int)(CANONICAL_CANVAS_W * scale_fit);
        int dest_h = (int)(CANONICAL_CANVAS_H * scale_fit);
        int dest_x = (win_w - dest_w) / 2;
        int dest_y = (win_h - dest_h) / 2;

        HDC hdc_screen = GetDC(win->hwnd);

        /* Clear black letterbox bars if needed */
        if (dest_x > 0 || dest_y > 0) {
            HBRUSH black_br = (HBRUSH)GetStockObject(BLACK_BRUSH);
            if (dest_x > 0) {
                RECT r_left = {0, 0, dest_x, win_h};
                RECT r_right = {dest_x + dest_w, 0, win_w, win_h};
                FillRect(hdc_screen, &r_left, black_br);
                FillRect(hdc_screen, &r_right, black_br);
            }
            if (dest_y > 0) {
                RECT r_top = {0, 0, win_w, dest_y};
                RECT r_bot = {0, dest_y + dest_h, win_w, win_h};
                FillRect(hdc_screen, &r_top, black_br);
                FillRect(hdc_screen, &r_bot, black_br);
            }
        }

        SetStretchBltMode(hdc_screen, HALFTONE);
        SetBrushOrgEx(hdc_screen, 0, 0, NULL);
        StretchBlt(
            hdc_screen,
            dest_x, dest_y, dest_w, dest_h,
            win->hdc_mem,
            0, 0, CANONICAL_CANVAS_W, CANONICAL_CANVAS_H,
            SRCCOPY
        );

        ReleaseDC(win->hwnd, hdc_screen);
    }

#endif
}
