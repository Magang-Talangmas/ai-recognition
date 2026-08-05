#ifndef ATTENDANCE_FACE_ENGINE_H
#define ATTENDANCE_FACE_ENGINE_H

#include <stdbool.h>
#include <stdint.h>
#include "attendance/config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FACE_EMBEDDING_DIM 512
#define MAX_DETECTED_FACES 64

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
} FaceBBox;

typedef struct {
    float x[5];
    float y[5];
} FaceLandmarks5;

typedef struct {
    FaceBBox bbox;
    FaceLandmarks5 landmarks;
    float embedding[FACE_EMBEDDING_DIM];
    bool has_embedding;
    bool quality_ok;
    float blur_score;
    float detection_score;
} FaceResult;

typedef struct {
    uint8_t *data;      /* BGR interleaved pixel data */
    int width;
    int height;
    int channels;       /* usually 3 */
    int stride;         /* bytes per row */
} ImageBuffer;

typedef struct FaceEngine FaceEngine;

/* Initialize FaceEngine with detection and recognition models */
FaceEngine *face_engine_create(const FaceConfig *config);

/* Free FaceEngine resources */
void face_engine_destroy(FaceEngine *engine);

/* Detect faces in a BGR image buffer. Returns number of faces detected (up to max_faces). */
int face_engine_detect(
    FaceEngine *engine,
    const ImageBuffer *frame,
    FaceResult *results,
    int max_faces
);

/* Extract 512-D ArcFace embedding for an aligned face crop (112x112 BGR) */
int face_engine_extract_embedding(
    FaceEngine *engine,
    const ImageBuffer *aligned_face,
    float *embedding_out
);

/* Align 5-point face to standard 112x112 ArcFace crop using similarity transform (Umeyama) */
int face_engine_align_face(
    const ImageBuffer *src_img,
    const FaceLandmarks5 *landmarks,
    ImageBuffer *dst_crop_112x112
);

/* Calculate Laplacian variance of image crop for blur assessment */
float face_engine_calculate_blur_score(const ImageBuffer *image, const FaceBBox *crop_box);

/* Build average L2-normalized template from multiple photos of a person */
int face_engine_build_template(
    FaceEngine *engine,
    const ImageBuffer *images,
    int num_images,
    float *template_out,
    int *valid_count_out
);

/* Utility to normalize a 512-d vector to unit L2 length in-place */
void face_vector_l2_normalize(float *vector, int dim);

#ifdef __cplusplus
}
#endif

#endif /* ATTENDANCE_FACE_ENGINE_H */
