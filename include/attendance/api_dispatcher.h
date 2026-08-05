#ifndef ATTENDANCE_API_DISPATCHER_H
#define ATTENDANCE_API_DISPATCHER_H

#include <stdbool.h>
#include <stdint.h>
#include "attendance/config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BackendDispatcher BackendDispatcher;

/* Initialize asynchronous HTTP background dispatcher */
BackendDispatcher *backend_dispatcher_create(const BackendConfig *config);

/* Destroy dispatcher and flush/stop worker thread */
void backend_dispatcher_destroy(BackendDispatcher *dispatcher);

/* Enqueue a CHECK_IN attendance event to be dispatched asynchronously */
bool backend_dispatcher_dispatch_checkin(
    BackendDispatcher *dispatcher,
    const char *employee_id,
    float similarity,
    const char *camera_id,
    const char *event_id,
    const char *detected_at
);

/* Perform a synchronous HTTP POST or GET request (used in sync CLI) */
int http_client_post_json(
    const char *url,
    const char *api_key,
    const char *json_body,
    float timeout_seconds,
    char *out_response,
    size_t out_buf_size
);

int http_client_get(
    const char *url,
    const char *auth_basic_user,
    const char *auth_basic_pass,
    float timeout_seconds,
    char *out_response,
    size_t out_buf_size
);

#ifdef __cplusplus
}
#endif

#endif /* ATTENDANCE_API_DISPATCHER_H */
