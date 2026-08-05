#ifndef ATTENDANCE_CONFIG_H
#define ATTENDANCE_CONFIG_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char source[512];
    char camera_id[128];
    int process_every_n_frames;
    int reconnect_delay_seconds;
    bool show_preview;
} CameraConfig;

typedef struct {
    char model_path[512];
    float confidence;
    char tracker[128];
    int bbox_padding;
} PersonConfig;

typedef struct {
    char model_pack[128];
    char model_root[512];
    char backend[64];         /* "onnxruntime" | "openvino" */
    char openvino_device[64];  /* "AUTO", "CPU", "GPU" */
    int detection_size;
    float detection_threshold;
    int min_face_size;
    float blur_threshold;
    float match_threshold;
    float match_margin;
    int vote_window;
    int votes_required;
} FaceConfig;

typedef struct {
    float line_y_ratio;
    bool inside_is_below_line;
    int duplicate_cooldown_seconds;
    char event_database[512];
    char embedding_file[512];
    bool allow_insecure_no_liveness;
} AttendanceConfig;

typedef struct {
    char api_url[512];
    char api_key[256];
    bool enabled;
    float timeout;
} BackendConfig;

typedef struct {
    CameraConfig camera;
    PersonConfig person;
    FaceConfig face;
    AttendanceConfig attendance;
    BackendConfig backend;
} AppConfig;

/* Load environment variables from .env file */
void config_load_dotenv(const char *env_path);

/* Expand ${VAR} placeholders in text string using current environment */
void config_expand_env(const char *input, char *output, size_t output_size);

/* Load application configuration from YAML/JSON file or defaults */
int config_load(const char *config_path, AppConfig *config);

/* Print active configuration to stdout */
void config_print(const AppConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* ATTENDANCE_CONFIG_H */
