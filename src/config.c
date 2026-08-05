#include "attendance/config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

static char *trim_whitespace(char *str) {
    if (!str) return NULL;
    while (isspace((unsigned char)*str)) str++;
    if (*str == 0) return str;
    char *end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
    return str;
}

static char *strip_quotes(char *str) {
    if (!str) return NULL;
    size_t len = strlen(str);
    if (len >= 2 && ((str[0] == '"' && str[len - 1] == '"') ||
                     (str[0] == '\'' && str[len - 1] == '\''))) {
        str[len - 1] = '\0';
        return str + 1;
    }
    return str;
}

void config_expand_env(const char *input, char *output, size_t output_size) {
    if (!input || !output || output_size == 0) return;
    char temp_in[1024];
    strncpy(temp_in, input, sizeof(temp_in) - 1);
    temp_in[sizeof(temp_in) - 1] = '\0';

    /* Perform up to 4 recursive expansion passes for nested variables */
    for (int pass = 0; pass < 4; pass++) {
        char temp_out[1024] = {0};
        const char *p = temp_in;
        size_t out_idx = 0;
        bool had_expansion = false;

        while (*p && out_idx + 1 < sizeof(temp_out)) {
            if (p[0] == '$' && p[1] == '{') {
                const char *end = strchr(p + 2, '}');
                if (end) {
                    char var_name[128];
                    size_t var_len = (size_t)(end - (p + 2));
                    if (var_len < sizeof(var_name)) {
                        strncpy(var_name, p + 2, var_len);
                        var_name[var_len] = '\0';

                        const char *val = getenv(var_name);
                        if (val) {
                            had_expansion = true;
                            size_t val_len = strlen(val);
                            for (size_t i = 0; i < val_len && out_idx + 1 < sizeof(temp_out); i++) {
                                temp_out[out_idx++] = val[i];
                            }
                        }
                        p = end + 1;
                        continue;
                    }
                }
            }
            temp_out[out_idx++] = *p++;
        }
        temp_out[out_idx] = '\0';
        strncpy(temp_in, temp_out, sizeof(temp_in) - 1);
        temp_in[sizeof(temp_in) - 1] = '\0';
        if (!had_expansion || !strstr(temp_in, "${")) break;
    }

    strncpy(output, temp_in, output_size - 1);
    output[output_size - 1] = '\0';
}

void config_load_dotenv(const char *env_path) {
    const char *candidates[] = {
        env_path,
        ".env",
        "../.env",
        "../ai-recognition/.env"
    };

    for (size_t c = 0; c < sizeof(candidates)/sizeof(candidates[0]); c++) {
        const char *path = candidates[c];
        if (!path) continue;
        FILE *f = fopen(path, "r");
        if (!f) continue;

        char line[1024];
        while (fgets(line, sizeof(line), f)) {
            char *trimmed = trim_whitespace(line);
            if (*trimmed == '#' || *trimmed == '\0') continue;

            char *eq = strchr(trimmed, '=');
            if (!eq) continue;

            *eq = '\0';
            char *key = trim_whitespace(trimmed);
            char *val = trim_whitespace(eq + 1);
            val = strip_quotes(val);

            if (*key) {
                char expanded_val[1024];
                config_expand_env(val, expanded_val, sizeof(expanded_val));

#ifdef _WIN32
                SetEnvironmentVariableA(key, expanded_val);
                _putenv_s(key, expanded_val);
#else
                setenv(key, expanded_val, 1);
#endif
            }
        }
        fclose(f);
        break;
    }
}

static void set_default_config(AppConfig *cfg) {
    memset(cfg, 0, sizeof(AppConfig));

    /* Camera defaults */
    snprintf(cfg->camera.source, sizeof(cfg->camera.source), "0");
    snprintf(cfg->camera.camera_id, sizeof(cfg->camera.camera_id), "main-entrance");
    cfg->camera.process_every_n_frames = 10;
    cfg->camera.reconnect_delay_seconds = 3;
    cfg->camera.show_preview = true;

    /* Person Detection defaults */
    snprintf(cfg->person.model_path, sizeof(cfg->person.model_path), "models/yolov10n.onnx");
    cfg->person.confidence = 0.40f;
    snprintf(cfg->person.tracker, sizeof(cfg->person.tracker), "bytetrack");
    cfg->person.bbox_padding = 0;

    /* Face defaults */
    snprintf(cfg->face.model_pack, sizeof(cfg->face.model_pack), "buffalo_l");
    snprintf(cfg->face.model_root, sizeof(cfg->face.model_root), ".insightface");
    snprintf(cfg->face.backend, sizeof(cfg->face.backend), "openvino");
    snprintf(cfg->face.openvino_device, sizeof(cfg->face.openvino_device), "AUTO");
    cfg->face.detection_size = 640;
    cfg->face.detection_threshold = 0.50f;
    cfg->face.min_face_size = 10;
    cfg->face.blur_threshold = 30.0f;
    cfg->face.match_threshold = 0.25f;
    cfg->face.match_margin = 0.07f;
    cfg->face.vote_window = 5;
    cfg->face.votes_required = 3;

    /* Attendance defaults */
    cfg->attendance.line_y_ratio = 0.62f;
    cfg->attendance.inside_is_below_line = true;
    cfg->attendance.duplicate_cooldown_seconds = 300;
    snprintf(cfg->attendance.event_database, sizeof(cfg->attendance.event_database), "data/attendance.db");
    snprintf(cfg->attendance.embedding_file, sizeof(cfg->attendance.embedding_file), "data/embeddings.bin");
    cfg->attendance.allow_insecure_no_liveness = true;

    /* Backend defaults */
    snprintf(cfg->backend.api_url, sizeof(cfg->backend.api_url), "http://localhost:3000/api/v1/attendance");
    snprintf(cfg->backend.api_key, sizeof(cfg->backend.api_key), "your-ml-api-key");
    cfg->backend.enabled = true;
    cfg->backend.timeout = 4.0f;
}

int config_load(const char *config_path, AppConfig *config) {
    if (!config) return -1;
    set_default_config(config);
    config_load_dotenv(".env");

    const char *path = config_path ? config_path : "config.yaml";
    FILE *f = fopen(path, "r");
    if (!f) {
        /* Fallback check in current directory */
        return 0;
    }

    char line[1024];
    char current_section[64] = "";

    while (fgets(line, sizeof(line), f)) {
        char *trimmed = trim_whitespace(line);
        if (*trimmed == '#' || *trimmed == '\0') continue;

        /* Check for section header (e.g. "camera:", "face:") */
        char *colon = strchr(trimmed, ':');
        if (!colon) continue;

        /* If line has no leading spaces and ends with ':', it's a section header */
        if (line[0] != ' ' && line[0] != '\t' && colon == trimmed + strlen(trimmed) - 1) {
            *colon = '\0';
            strncpy(current_section, trim_whitespace(trimmed), sizeof(current_section) - 1);
            continue;
        }

        /* Key-Value pair within current section */
        *colon = '\0';
        char *key = trim_whitespace(trimmed);
        char *raw_val = trim_whitespace(colon + 1);

        /* Remove inline comments if present */
        char *hash = strchr(raw_val, '#');
        if (hash) {
            *hash = '\0';
            raw_val = trim_whitespace(raw_val);
        }
        raw_val = strip_quotes(raw_val);

        char val[512];
        config_expand_env(raw_val, val, sizeof(val));

        /* Camera Section */
        if (strcmp(current_section, "camera") == 0) {
            if (strcmp(key, "source") == 0 && val[0]) strncpy(config->camera.source, val, sizeof(config->camera.source) - 1);
            else if (strcmp(key, "camera_id") == 0 && val[0]) strncpy(config->camera.camera_id, val, sizeof(config->camera.camera_id) - 1);
            else if (strcmp(key, "process_every_n_frames") == 0 && val[0]) config->camera.process_every_n_frames = atoi(val);
            else if (strcmp(key, "reconnect_delay_seconds") == 0 && val[0]) config->camera.reconnect_delay_seconds = atoi(val);
            else if (strcmp(key, "show_preview") == 0 && val[0]) config->camera.show_preview = (strcmp(val, "true") == 0 || strcmp(val, "1") == 0);
        }
        /* Person Detection Section */
        else if (strcmp(current_section, "person_detection") == 0) {
            if (strcmp(key, "model_path") == 0 && val[0]) strncpy(config->person.model_path, val, sizeof(config->person.model_path) - 1);
            else if (strcmp(key, "confidence") == 0 && val[0]) config->person.confidence = (float)atof(val);
            else if (strcmp(key, "tracker") == 0 && val[0]) strncpy(config->person.tracker, val, sizeof(config->person.tracker) - 1);
            else if (strcmp(key, "bbox_padding") == 0 && val[0]) config->person.bbox_padding = atoi(val);
        }
        /* Face Section */
        else if (strcmp(current_section, "face") == 0) {
            if (strcmp(key, "model_pack") == 0 && val[0]) strncpy(config->face.model_pack, val, sizeof(config->face.model_pack) - 1);
            else if (strcmp(key, "model_root") == 0 && val[0]) strncpy(config->face.model_root, val, sizeof(config->face.model_root) - 1);
            else if (strcmp(key, "backend") == 0 && val[0]) strncpy(config->face.backend, val, sizeof(config->face.backend) - 1);
            else if (strcmp(key, "openvino_device") == 0 && val[0]) strncpy(config->face.openvino_device, val, sizeof(config->face.openvino_device) - 1);
            else if (strcmp(key, "detection_size") == 0 && val[0]) config->face.detection_size = atoi(val);
            else if (strcmp(key, "detection_threshold") == 0 && val[0]) config->face.detection_threshold = (float)atof(val);
            else if (strcmp(key, "min_face_size") == 0 && val[0]) config->face.min_face_size = atoi(val);
            else if (strcmp(key, "blur_threshold") == 0 && val[0]) config->face.blur_threshold = (float)atof(val);
            else if (strcmp(key, "match_threshold") == 0 && val[0]) config->face.match_threshold = (float)atof(val);
            else if (strcmp(key, "match_margin") == 0 && val[0]) config->face.match_margin = (float)atof(val);
            else if (strcmp(key, "vote_window") == 0 && val[0]) config->face.vote_window = atoi(val);
            else if (strcmp(key, "votes_required") == 0 && val[0]) config->face.votes_required = atoi(val);
        }
        /* Attendance Section */
        else if (strcmp(current_section, "attendance") == 0) {
            if (strcmp(key, "line_y_ratio") == 0 && val[0]) config->attendance.line_y_ratio = (float)atof(val);
            else if (strcmp(key, "inside_is_below_line") == 0 && val[0]) config->attendance.inside_is_below_line = (strcmp(val, "true") == 0 || strcmp(val, "1") == 0);
            else if (strcmp(key, "duplicate_cooldown_seconds") == 0 && val[0]) config->attendance.duplicate_cooldown_seconds = atoi(val);
            else if (strcmp(key, "event_database") == 0 && val[0]) strncpy(config->attendance.event_database, val, sizeof(config->attendance.event_database) - 1);
            else if (strcmp(key, "embedding_file") == 0 && val[0]) strncpy(config->attendance.embedding_file, val, sizeof(config->attendance.embedding_file) - 1);
            else if (strcmp(key, "allow_insecure_no_liveness") == 0 && val[0]) config->attendance.allow_insecure_no_liveness = (strcmp(val, "true") == 0 || strcmp(val, "1") == 0);
        }
        /* Backend Section */
        else if (strcmp(current_section, "backend") == 0) {
            if (strcmp(key, "api_url") == 0 && val[0]) strncpy(config->backend.api_url, val, sizeof(config->backend.api_url) - 1);
            else if (strcmp(key, "api_key") == 0 && val[0]) strncpy(config->backend.api_key, val, sizeof(config->backend.api_key) - 1);
            else if (strcmp(key, "enabled") == 0 && val[0]) config->backend.enabled = (strcmp(val, "true") == 0 || strcmp(val, "1") == 0);
            else if (strcmp(key, "timeout") == 0 && val[0]) config->backend.timeout = (float)atof(val);
        }
    }

    fclose(f);
    return 0;
}

void config_print(const AppConfig *cfg) {
    printf("====================================================\n");
    printf("           ACTIVE CONFIGURATION                     \n");
    printf("====================================================\n");
    printf("[Camera] source: %s | ID: %s | skip: %d | preview: %s\n",
           cfg->camera.source, cfg->camera.camera_id,
           cfg->camera.process_every_n_frames,
           cfg->camera.show_preview ? "true" : "false");
    printf("[Face] backend: %s | device: %s | size: %d | det_thresh: %.2f\n",
           cfg->face.backend, cfg->face.openvino_device,
           cfg->face.detection_size, cfg->face.detection_threshold);
    printf("       match_thresh: %.2f | margin: %.2f | votes: %d/%d\n",
           cfg->face.match_threshold, cfg->face.match_margin,
           cfg->face.votes_required, cfg->face.vote_window);
    printf("[Attendance] db: %s | embeddings: %s | cooldown: %ds\n",
           cfg->attendance.event_database, cfg->attendance.embedding_file,
           cfg->attendance.duplicate_cooldown_seconds);
    printf("[Backend] enabled: %s | URL: %s\n",
           cfg->backend.enabled ? "true" : "false", cfg->backend.api_url);
    printf("====================================================\n");
}
