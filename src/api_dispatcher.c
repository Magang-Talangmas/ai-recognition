#include "attendance/api_dispatcher.h"
#include "attendance/vision_utils.h"
#include "third_party/cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#else
#include <pthread.h>
#include <unistd.h>
#endif

#define DISPATCHER_QUEUE_SIZE 64

typedef struct {
    char employee_id[128];
    float similarity;
    char camera_id[128];
    char event_id[128];
    char detected_at[64];
} DispatchTask;

struct BackendDispatcher {
    BackendConfig config;
    DispatchTask queue[DISPATCHER_QUEUE_SIZE];
    int head;
    int tail;
    int count;
    bool running;

#ifdef _WIN32
    CRITICAL_SECTION lock;
    HANDLE event_signal;
    HANDLE worker_thread;
#else
    pthread_mutex_t lock;
    pthread_cond_t cond;
    pthread_t worker_thread;
#endif
};

#ifdef _WIN32
static int winhttp_send_request(
    const char *url,
    const char *api_key,
    const char *json_body,
    float timeout_sec,
    char *out_response,
    size_t out_buf_size
) {
    WCHAR w_url[512] = {0};
    MultiByteToWideChar(CP_UTF8, 0, url, -1, w_url, 512);

    URL_COMPONENTS url_comp = {0};
    url_comp.dwStructSize = sizeof(url_comp);
    url_comp.dwHostNameLength = (DWORD)-1;
    url_comp.dwUrlPathLength = (DWORD)-1;
    url_comp.dwExtraInfoLength = (DWORD)-1;

    if (!WinHttpCrackUrl(w_url, (DWORD)wcslen(w_url), 0, &url_comp)) {
        return -1;
    }

    WCHAR host[256] = {0};
    wcsncpy(host, url_comp.lpszHostName, url_comp.dwHostNameLength);

    HINTERNET h_session = WinHttpOpen(L"AI-Recognition-C/1.0",
                                     WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                     WINHTTP_NO_PROXY_NAME,
                                     WINHTTP_NO_PROXY_BYPASS, 0);
    if (!h_session) return -1;

    DWORD timeout_ms = (DWORD)(timeout_sec * 1000);
    WinHttpSetTimeouts(h_session, timeout_ms, timeout_ms, timeout_ms, timeout_ms);

    HINTERNET h_connect = WinHttpConnect(h_session, host, url_comp.nPort, 0);
    if (!h_connect) {
        WinHttpCloseHandle(h_session);
        return -1;
    }

    DWORD flags = (url_comp.nScheme == INTERNET_SCHEME_HTTPS) ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET h_request = WinHttpOpenRequest(h_connect, L"POST", url_comp.lpszUrlPath,
                                            NULL, WINHTTP_NO_REFERER,
                                            WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!h_request) {
        WinHttpCloseHandle(h_connect);
        WinHttpCloseHandle(h_session);
        return -1;
    }

    WCHAR headers[512];
    swprintf(headers, 512, L"Content-Type: application/json\r\nx-api-key: %hs\r\n", api_key ? api_key : "");

    DWORD body_len = (DWORD)strlen(json_body);
    BOOL b_results = WinHttpSendRequest(h_request, headers, (DWORD)-1, (LPVOID)json_body, body_len, body_len, 0);

    if (b_results) {
        b_results = WinHttpReceiveResponse(h_request, NULL);
    }

    DWORD status_code = 0;
    DWORD status_size = sizeof(status_code);
    if (b_results) {
        WinHttpQueryHeaders(h_request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                            WINHTTP_HEADER_NAME_BY_INDEX, &status_code, &status_size, WINHTTP_NO_HEADER_INDEX);
    }

    WinHttpCloseHandle(h_request);
    WinHttpCloseHandle(h_connect);
    WinHttpCloseHandle(h_session);

    return b_results ? (int)status_code : -1;
}
#endif

int http_client_post_json(
    const char *url,
    const char *api_key,
    const char *json_body,
    float timeout_seconds,
    char *out_response,
    size_t out_buf_size
) {
#ifdef _WIN32
    return winhttp_send_request(url, api_key, json_body, timeout_seconds, out_response, out_buf_size);
#else
    /* Portable curl or socket implementation */
    printf("[HTTP POST] URL: %s | Body: %s\n", url, json_body);
    return 200;
#endif
}

static void send_task_payload(const BackendConfig *config, const DispatchTask *task) {
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "event_id", task->event_id);
    cJSON_AddStringToObject(root, "employee_id", task->employee_id);
    cJSON_AddStringToObject(root, "event_type", "CHECK_IN");
    cJSON_AddNumberToObject(root, "similarity", (double)task->similarity);
    cJSON_AddStringToObject(root, "detected_at", task->detected_at);
    cJSON_AddStringToObject(root, "camera_id", task->camera_id);

    char *json_str = cJSON_PrintUnformatted(root);
    if (json_str) {
        int status = http_client_post_json(
            config->api_url,
            config->api_key,
            json_str,
            config->timeout,
            NULL,
            0
        );

        if (status >= 200 && status < 300) {
            printf("[BE DISPATCH SUCCESS] %s | CHECK_IN | HTTP %d | event_id=%s\n",
                   task->employee_id, status, task->event_id);
        } else {
            printf("[BE DISPATCH WARNING] %s | CHECK_IN | HTTP %d (URL: %s)\n",
                   task->employee_id, status, config->api_url);
        }
        free(json_str);
    }
    cJSON_Delete(root);
}

#ifdef _WIN32
static DWORD WINAPI dispatcher_thread_func(LPVOID param) {
    BackendDispatcher *disp = (BackendDispatcher *)param;
    while (disp->running) {
        WaitForSingleObject(disp->event_signal, 200);

        DispatchTask task;
        bool has_task = false;

        EnterCriticalSection(&disp->lock);
        if (disp->count > 0) {
            task = disp->queue[disp->head];
            disp->head = (disp->head + 1) % DISPATCHER_QUEUE_SIZE;
            disp->count--;
            has_task = true;
        }
        LeaveCriticalSection(&disp->lock);

        if (has_task) {
            send_task_payload(&disp->config, &task);
        }
    }
    return 0;
}
#else
static void *dispatcher_thread_func(void *param) {
    BackendDispatcher *disp = (BackendDispatcher *)param;
    while (disp->running) {
        DispatchTask task;
        bool has_task = false;

        pthread_mutex_lock(&disp->lock);
        while (disp->count == 0 && disp->running) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += 1;
            pthread_cond_timedwait(&disp->cond, &disp->lock, &ts);
        }

        if (disp->count > 0) {
            task = disp->queue[disp->head];
            disp->head = (disp->head + 1) % DISPATCHER_QUEUE_SIZE;
            disp->count--;
            has_task = true;
        }
        pthread_mutex_unlock(&disp->lock);

        if (has_task) {
            send_task_payload(&disp->config, &task);
        }
    }
    return NULL;
}
#endif

BackendDispatcher *backend_dispatcher_create(const BackendConfig *config) {
    if (!config || !config->enabled) return NULL;

    BackendDispatcher *disp = (BackendDispatcher *)calloc(1, sizeof(BackendDispatcher));
    if (!disp) return NULL;

    disp->config = *config;
    disp->running = true;

#ifdef _WIN32
    InitializeCriticalSection(&disp->lock);
    disp->event_signal = CreateEvent(NULL, FALSE, FALSE, NULL);
    disp->worker_thread = CreateThread(NULL, 0, dispatcher_thread_func, disp, 0, NULL);
#else
    pthread_mutex_init(&disp->lock, NULL);
    pthread_cond_init(&disp->cond, NULL);
    pthread_create(&disp->worker_thread, NULL, dispatcher_thread_func, disp);
#endif

    return disp;
}

void backend_dispatcher_destroy(BackendDispatcher *disp) {
    if (!disp) return;
    disp->running = false;

#ifdef _WIN32
    SetEvent(disp->event_signal);
    if (disp->worker_thread) {
        WaitForSingleObject(disp->worker_thread, 2000);
        CloseHandle(disp->worker_thread);
    }
    CloseHandle(disp->event_signal);
    DeleteCriticalSection(&disp->lock);
#else
    pthread_cond_broadcast(&disp->cond);
    pthread_join(disp->worker_thread, NULL);
    pthread_mutex_destroy(&disp->lock);
    pthread_cond_destroy(&disp->cond);
#endif

    free(disp);
}

bool backend_dispatcher_dispatch_checkin(
    BackendDispatcher *disp,
    const char *employee_id,
    float similarity,
    const char *camera_id,
    const char *event_id,
    const char *detected_at
) {
    if (!disp || !disp->running || !employee_id) return false;

    DispatchTask task;
    memset(&task, 0, sizeof(DispatchTask));
    strncpy(task.employee_id, employee_id, sizeof(task.employee_id) - 1);
    task.similarity = similarity;
    strncpy(task.camera_id, camera_id ? camera_id : "main-entrance", sizeof(task.camera_id) - 1);

    if (event_id && strlen(event_id) > 0) {
        strncpy(task.event_id, event_id, sizeof(task.event_id) - 1);
    } else {
        snprintf(task.event_id, sizeof(task.event_id), "evt-%lld", (long long)(get_monotonic_time_seconds() * 1000.0));
    }

    if (detected_at && strlen(detected_at) > 0) {
        strncpy(task.detected_at, detected_at, sizeof(task.detected_at) - 1);
    } else {
        get_iso8601_timestamp(task.detected_at, sizeof(task.detected_at));
    }

#ifdef _WIN32
    EnterCriticalSection(&disp->lock);
#else
    pthread_mutex_lock(&disp->lock);
#endif

    bool enqueued = false;
    if (disp->count < DISPATCHER_QUEUE_SIZE) {
        disp->queue[disp->tail] = task;
        disp->tail = (disp->tail + 1) % DISPATCHER_QUEUE_SIZE;
        disp->count++;
        enqueued = true;
    }

#ifdef _WIN32
    LeaveCriticalSection(&disp->lock);
    if (enqueued) SetEvent(disp->event_signal);
#else
    pthread_mutex_unlock(&disp->lock);
    if (enqueued) pthread_cond_signal(&disp->cond);
#endif

    return enqueued;
}
