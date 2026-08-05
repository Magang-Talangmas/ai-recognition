/*
  Copyright (c) 2009-2017 Dave Gamble and cJSON contributors

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#ifndef cJSON__h
#define cJSON__h

#ifdef __cplusplus
extern "C"
{
#endif

#if !defined(__WINDOWS__) && (defined(WIN32) || defined(WIN64) || defined(_MSC_VER) || defined(_WIN32))
#define __WINDOWS__
#endif

#ifdef __WINDOWS__
#define CJSON_CDECL __cdecl
#define CJSON_STDCALL __stdcall
#else
#define CJSON_CDECL
#define CJSON_STDCALL
#endif

#include <stddef.h>

#define cJSON_Invalid (0)
#define cJSON_False  (1 << 0)
#define cJSON_True   (1 << 1)
#define cJSON_NULL   (1 << 2)
#define cJSON_Number (1 << 3)
#define cJSON_String (1 << 4)
#define cJSON_Array  (1 << 5)
#define cJSON_Object (1 << 6)
#define cJSON_Raw    (1 << 7)

#define cJSON_IsReference 256
#define cJSON_StringIsConst 512

typedef struct cJSON
{
    struct cJSON *next;
    struct cJSON *prev;
    struct cJSON *child;
    int type;
    char *valuestring;
    int valueint;
    double valuedouble;
    char *string;
} cJSON;

typedef struct cJSON_Hooks
{
      void *(CJSON_CDECL *malloc_fn)(size_t sz);
      void (CJSON_CDECL *free_fn)(void *ptr);
} cJSON_Hooks;

#define cJSON_SetIntValue(object, number) ((object) ? (object)->valueint = (object)->valuedouble = (number) : (number))
#define cJSON_SetNumberValue(object, number) ((object) ? (object)->valueint = (int)((object)->valuedouble = (number)) : (number))

extern const char *cJSON_Version(void);
extern void cJSON_InitHooks(cJSON_Hooks* hooks);
extern cJSON *cJSON_Parse(const char *value);
extern cJSON *cJSON_ParseWithLength(const char *value, size_t buffer_length);
extern cJSON *cJSON_ParseWithOpts(const char *value, const char **return_parse_end, int require_null_terminated);
extern cJSON *cJSON_ParseWithLengthOpts(const char *value, size_t buffer_length, const char **return_parse_end, int require_null_terminated);
extern char  *cJSON_Print(const cJSON *item);
extern char  *cJSON_PrintUnformatted(const cJSON *item);
extern char  *cJSON_PrintBuffered(const cJSON *item, int prebuffer, int fmt);
extern int    cJSON_PrintPreallocated(cJSON *item, char *buffer, const int length, const int format);
extern void   cJSON_Delete(cJSON *c);

extern int cJSON_GetArraySize(const cJSON *array);
extern cJSON *cJSON_GetArrayItem(const cJSON *array, int index);
extern cJSON *cJSON_GetObjectItem(const cJSON * const object, const char * const string);
extern cJSON *cJSON_GetObjectItemCaseSensitive(const cJSON * const object, const char * const string);
extern int cJSON_HasObjectItem(const cJSON *object, const char *string);
extern const char *cJSON_GetErrorPtr(void);

extern int cJSON_IsInvalid(const cJSON * const item);
extern int cJSON_IsFalse(const cJSON * const item);
extern int cJSON_IsTrue(const cJSON * const item);
extern int cJSON_IsBool(const cJSON * const item);
extern int cJSON_IsNull(const cJSON * const item);
extern int cJSON_IsNumber(const cJSON * const item);
extern int cJSON_IsString(const cJSON * const item);
extern int cJSON_IsArray(const cJSON * const item);
extern int cJSON_IsObject(const cJSON * const item);
extern int cJSON_IsRaw(const cJSON * const item);

extern cJSON *cJSON_CreateNull(void);
extern cJSON *cJSON_CreateTrue(void);
extern cJSON *cJSON_CreateFalse(void);
extern cJSON *cJSON_CreateBool(int boolean);
extern cJSON *cJSON_CreateNumber(double num);
extern cJSON *cJSON_CreateString(const char *string);
extern cJSON *cJSON_CreateRaw(const char *raw);
extern cJSON *cJSON_CreateArray(void);
extern cJSON *cJSON_CreateObject(void);

extern cJSON *cJSON_CreateIntArray(const int *numbers, int count);
extern cJSON *cJSON_CreateFloatArray(const float *numbers, int count);
extern cJSON *cJSON_CreateDoubleArray(const double *numbers, int count);
extern cJSON *cJSON_CreateStringArray(const char *const *strings, int count);

extern int cJSON_AddItemToArray(cJSON *array, cJSON *item);
extern int cJSON_AddItemToObject(cJSON *object, const char *string, cJSON *item);
extern int cJSON_AddItemToObjectCS(cJSON *object, const char *string, cJSON *item);
extern int cJSON_AddItemReferenceToArray(cJSON *array, cJSON *item);
extern int cJSON_AddItemReferenceToObject(cJSON *object, const char *string, cJSON *item);

extern cJSON *cJSON_DetachItemViaPointer(cJSON *parent, cJSON * const item);
extern cJSON *cJSON_DetachItemFromArray(cJSON *array, int which);
extern void cJSON_DeleteItemFromArray(cJSON *array, int which);
extern cJSON *cJSON_DetachItemFromObject(cJSON *object, const char *string);
extern cJSON *cJSON_DetachItemFromObjectCaseSensitive(cJSON *object, const char *string);
extern void cJSON_DeleteItemFromObject(cJSON *object, const char *string);
extern void cJSON_DeleteItemFromObjectCaseSensitive(cJSON *object, const char *string);

extern int cJSON_InsertItemInArray(cJSON *array, int which, cJSON *newitem);
extern int cJSON_ReplaceItemViaPointer(cJSON * const parent, cJSON * const item, cJSON * replacement);
extern int cJSON_ReplaceItemInArray(cJSON *array, int which, cJSON *newitem);
extern int cJSON_ReplaceItemInObject(cJSON *object,const char *string,cJSON *newitem);
extern int cJSON_ReplaceItemInObjectCaseSensitive(cJSON *object,const char *string,cJSON *newitem);

extern cJSON *cJSON_Duplicate(const cJSON *item, int recurse);
extern int cJSON_Compare(const cJSON * const a, const cJSON * const b, const int case_sensitive);
extern void cJSON_Minify(char *json);

extern cJSON *cJSON_AddNullToObject(cJSON * const object, const char * const name);
extern cJSON *cJSON_AddTrueToObject(cJSON * const object, const char * const name);
extern cJSON *cJSON_AddFalseToObject(cJSON * const object, const char * const name);
extern cJSON *cJSON_AddBoolToObject(cJSON * const object, const char * const name, const int boolean);
extern cJSON *cJSON_AddNumberToObject(cJSON * const object, const char * const name, const double number);
extern cJSON *cJSON_AddStringToObject(cJSON * const object, const char * const name, const char * const string);
extern cJSON *cJSON_AddRawToObject(cJSON * const object, const char * const name, const char * const raw);
extern cJSON *cJSON_AddObjectToObject(cJSON * const object, const char * const name);
extern cJSON *cJSON_AddArrayToObject(cJSON * const object, const char * const name);

#define cJSON_SetArrayItem(array, which, newitem) cJSON_ReplaceItemInArray(array, which, newitem)
#define cJSON_SetObjectItem(object, string, newitem) cJSON_ReplaceItemInObject(object, string, newitem)

#ifdef __cplusplus
}
#endif

#endif
