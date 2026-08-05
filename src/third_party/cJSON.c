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

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <ctype.h>
#include <float.h>

#include "third_party/cJSON.h"

static struct internal_hooks
{
    void *(CJSON_CDECL *allocate)(size_t size);
    void (CJSON_CDECL *deallocate)(void *pointer);
    void *(CJSON_CDECL *reallocate)(void *pointer, size_t size);
} global_hooks = { malloc, free, realloc };

static unsigned char* cJSON_strdup(const unsigned char* string, const struct internal_hooks * const hooks)
{
    size_t length = 0;
    unsigned char *copy = NULL;

    if (string == NULL)
    {
        return NULL;
    }

    length = strlen((const char*)string) + sizeof("");
    copy = (unsigned char*)hooks->allocate(length);
    if (copy == NULL)
    {
        return NULL;
    }
    memcpy(copy, string, length);

    return copy;
}

void cJSON_InitHooks(cJSON_Hooks* hooks)
{
    if (hooks == NULL)
    {
        global_hooks.allocate = malloc;
        global_hooks.deallocate = free;
        global_hooks.reallocate = realloc;
        return;
    }

    global_hooks.allocate = malloc;
    if (hooks->malloc_fn != NULL)
    {
        global_hooks.allocate = hooks->malloc_fn;
    }

    global_hooks.deallocate = free;
    if (hooks->free_fn != NULL)
    {
        global_hooks.deallocate = hooks->free_fn;
    }

    global_hooks.reallocate = NULL;
}

static cJSON *cJSON_New_Item(const struct internal_hooks * const hooks)
{
    cJSON* node = (cJSON*)hooks->allocate(sizeof(cJSON));
    if (node)
    {
        memset(node, '\0', sizeof(cJSON));
    }

    return node;
}

void cJSON_Delete(cJSON *item)
{
    cJSON *next = NULL;
    while (item != NULL)
    {
        next = item->next;
        if (!(item->type & cJSON_IsReference) && (item->child != NULL))
        {
            cJSON_Delete(item->child);
        }
        if (!(item->type & cJSON_IsReference) && (item->valuestring != NULL))
        {
            global_hooks.deallocate(item->valuestring);
        }
        if (!(item->type & cJSON_StringIsConst) && (item->string != NULL))
        {
            global_hooks.deallocate(item->string);
        }
        global_hooks.deallocate(item);
        item = next;
    }
}

typedef struct
{
    const unsigned char *content;
    size_t length;
    size_t offset;
    size_t depth;
    struct internal_hooks hooks;
} parse_buffer;

#define can_read(buffer, size) ((buffer != NULL) && (((buffer)->offset + size) <= (buffer)->length))
#define cannot_read(buffer, size) (!can_read(buffer, size))
#define get_at_offset(buffer) ((buffer)->content + (buffer)->offset)

static parse_buffer *buffer_skip_whitespace(parse_buffer * const buffer)
{
    if ((buffer == NULL) || (buffer->content == NULL))
    {
        return NULL;
    }

    if (cannot_read(buffer, 1))
    {
        return buffer;
    }

    while (can_read(buffer, 1) && (get_at_offset(buffer)[0] <= 32))
    {
        buffer->offset++;
    }

    if (buffer->offset >= buffer->length)
    {
        buffer->offset = buffer->length;
    }

    return buffer;
}

static int parse_string(cJSON * const item, parse_buffer * const input_buffer)
{
    const unsigned char *input_pointer = get_at_offset(input_buffer) + 1;
    const unsigned char *input_end = get_at_offset(input_buffer) + 1;
    unsigned char *output_pointer = NULL;
    unsigned char *output = NULL;
    size_t allocation_length = 0;
    size_t skipped_bytes = 0;

    if (cannot_read(input_buffer, 2) || (get_at_offset(input_buffer)[0] != '\"'))
    {
        return 0;
    }

    while (((size_t)(input_end - input_buffer->content) < input_buffer->length) && (*input_end != '\"'))
    {
        if (input_end[0] == '\\')
        {
            skipped_bytes++;
            input_end++;
        }
        input_end++;
    }
    if (((size_t)(input_end - input_buffer->content) >= input_buffer->length) || (*input_end != '\"'))
    {
        return 0;
    }

    allocation_length = (size_t)(input_end - get_at_offset(input_buffer)) - skipped_bytes;
    output = (unsigned char*)input_buffer->hooks.allocate(allocation_length + sizeof(""));
    if (output == NULL)
    {
        return 0;
    }

    output_pointer = output;
    while (input_pointer < input_end)
    {
        if (*input_pointer != '\\')
        {
            *output_pointer++ = *input_pointer++;
        }
        else
        {
            unsigned char sequence_length = 2;
            if ((input_end - input_pointer) < 1)
            {
                input_buffer->hooks.deallocate(output);
                return 0;
            }
            switch (input_pointer[1])
            {
                case 'b': *output_pointer++ = '\b'; break;
                case 'f': *output_pointer++ = '\f'; break;
                case 'n': *output_pointer++ = '\n'; break;
                case 'r': *output_pointer++ = '\r'; break;
                case 't': *output_pointer++ = '\t'; break;
                case '\"':
                case '\\':
                case '/': *output_pointer++ = input_pointer[1]; break;
                default:
                    *output_pointer++ = input_pointer[1];
                    break;
            }
            input_pointer += sequence_length;
        }
    }
    *output_pointer = '\0';

    item->type = cJSON_String;
    item->valuestring = (char*)output;
    input_buffer->offset = (size_t)(input_end - input_buffer->content) + 1;
    return 1;
}

static int parse_number(cJSON * const item, parse_buffer * const input_buffer)
{
    double number = 0;
    unsigned char *after_end = NULL;
    const unsigned char * const cur = get_at_offset(input_buffer);

    if (cur == NULL) return 0;
    number = strtod((const char*)cur, (char**)&after_end);
    if (cur == after_end) return 0;

    item->valuedouble = number;
    item->valueint = (int)number;
    item->type = cJSON_Number;
    input_buffer->offset += (size_t)(after_end - cur);
    return 1;
}

static int parse_value(cJSON * const item, parse_buffer * const input_buffer);

static int parse_array(cJSON * const item, parse_buffer * const input_buffer)
{
    cJSON *head = NULL;
    cJSON *current_item = NULL;

    if (cannot_read(input_buffer, 1) || (get_at_offset(input_buffer)[0] != '['))
    {
        return 0;
    }
    input_buffer->offset++;
    buffer_skip_whitespace(input_buffer);
    if (can_read(input_buffer, 1) && (get_at_offset(input_buffer)[0] == ']'))
    {
        input_buffer->offset++;
        item->type = cJSON_Array;
        return 1;
    }

    current_item = cJSON_New_Item(&(input_buffer->hooks));
    if (current_item == NULL) return 0;
    item->child = head = current_item;

    if (!parse_value(current_item, input_buffer)) return 0;
    buffer_skip_whitespace(input_buffer);

    while (can_read(input_buffer, 1) && (get_at_offset(input_buffer)[0] == ','))
    {
        cJSON *new_item = NULL;
        input_buffer->offset++;
        buffer_skip_whitespace(input_buffer);
        new_item = cJSON_New_Item(&(input_buffer->hooks));
        if (new_item == NULL) return 0;
        current_item->next = new_item;
        new_item->prev = current_item;
        current_item = new_item;
        if (!parse_value(current_item, input_buffer)) return 0;
        buffer_skip_whitespace(input_buffer);
    }

    if (can_read(input_buffer, 1) && (get_at_offset(input_buffer)[0] == ']'))
    {
        input_buffer->offset++;
        item->type = cJSON_Array;
        return 1;
    }
    return 0;
}

static int parse_object(cJSON * const item, parse_buffer * const input_buffer)
{
    cJSON *head = NULL;
    cJSON *current_item = NULL;

    if (cannot_read(input_buffer, 1) || (get_at_offset(input_buffer)[0] != '{'))
    {
        return 0;
    }
    input_buffer->offset++;
    buffer_skip_whitespace(input_buffer);
    if (can_read(input_buffer, 1) && (get_at_offset(input_buffer)[0] == '}'))
    {
        input_buffer->offset++;
        item->type = cJSON_Object;
        return 1;
    }

    current_item = cJSON_New_Item(&(input_buffer->hooks));
    if (current_item == NULL) return 0;
    item->child = head = current_item;

    if (!parse_string(current_item, input_buffer)) return 0;
    buffer_skip_whitespace(input_buffer);
    current_item->string = current_item->valuestring;
    current_item->valuestring = NULL;

    if (cannot_read(input_buffer, 1) || (get_at_offset(input_buffer)[0] != ':')) return 0;
    input_buffer->offset++;
    buffer_skip_whitespace(input_buffer);

    if (!parse_value(current_item, input_buffer)) return 0;
    buffer_skip_whitespace(input_buffer);

    while (can_read(input_buffer, 1) && (get_at_offset(input_buffer)[0] == ','))
    {
        cJSON *new_item = NULL;
        input_buffer->offset++;
        buffer_skip_whitespace(input_buffer);
        new_item = cJSON_New_Item(&(input_buffer->hooks));
        if (new_item == NULL) return 0;
        current_item->next = new_item;
        new_item->prev = current_item;
        current_item = new_item;

        if (!parse_string(current_item, input_buffer)) return 0;
        buffer_skip_whitespace(input_buffer);
        current_item->string = current_item->valuestring;
        current_item->valuestring = NULL;

        if (cannot_read(input_buffer, 1) || (get_at_offset(input_buffer)[0] != ':')) return 0;
        input_buffer->offset++;
        buffer_skip_whitespace(input_buffer);

        if (!parse_value(current_item, input_buffer)) return 0;
        buffer_skip_whitespace(input_buffer);
    }

    if (can_read(input_buffer, 1) && (get_at_offset(input_buffer)[0] == '}'))
    {
        input_buffer->offset++;
        item->type = cJSON_Object;
        return 1;
    }
    return 0;
}

static int parse_value(cJSON * const item, parse_buffer * const input_buffer)
{
    if ((input_buffer == NULL) || (input_buffer->content == NULL)) return 0;

    buffer_skip_whitespace(input_buffer);
    if (cannot_read(input_buffer, 1)) return 0;

    switch (get_at_offset(input_buffer)[0])
    {
        case 'n':
            if (can_read(input_buffer, 4) && (strncmp((const char*)get_at_offset(input_buffer), "null", 4) == 0))
            {
                item->type = cJSON_NULL;
                input_buffer->offset += 4;
                return 1;
            }
            return 0;
        case 't':
            if (can_read(input_buffer, 4) && (strncmp((const char*)get_at_offset(input_buffer), "true", 4) == 0))
            {
                item->type = cJSON_True;
                item->valueint = 1;
                input_buffer->offset += 4;
                return 1;
            }
            return 0;
        case 'f':
            if (can_read(input_buffer, 5) && (strncmp((const char*)get_at_offset(input_buffer), "false", 5) == 0))
            {
                item->type = cJSON_False;
                item->valueint = 0;
                input_buffer->offset += 5;
                return 1;
            }
            return 0;
        case '\"':
            return parse_string(item, input_buffer);
        case '[':
            return parse_array(item, input_buffer);
        case '{':
            return parse_object(item, input_buffer);
        case '-':
        case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9':
            return parse_number(item, input_buffer);
        default:
            return 0;
    }
}

cJSON *cJSON_ParseWithLengthOpts(const char *value, size_t buffer_length, const char **return_parse_end, int require_null_terminated)
{
    parse_buffer buffer = { 0, 0, 0, 0, { 0, 0, 0 } };
    cJSON *item = NULL;

    if (value == NULL || buffer_length == 0) return NULL;
    buffer.content = (const unsigned char*)value;
    buffer.length = buffer_length;
    buffer.offset = 0;
    buffer.hooks = global_hooks;

    item = cJSON_New_Item(&global_hooks);
    if (item == NULL) return NULL;

    if (!parse_value(item, &buffer))
    {
        cJSON_Delete(item);
        return NULL;
    }

    if (require_null_terminated)
    {
        buffer_skip_whitespace(&buffer);
        if (buffer.offset < buffer.length)
        {
            cJSON_Delete(item);
            return NULL;
        }
    }

    if (return_parse_end)
    {
        *return_parse_end = (const char*)buffer.content + buffer.offset;
    }

    return item;
}

cJSON *cJSON_Parse(const char *value)
{
    return cJSON_ParseWithLengthOpts(value, value ? strlen(value) : 0, 0, 0);
}

typedef struct
{
    char *buffer;
    size_t length;
    size_t offset;
    size_t depth;
    int format;
} printbuffer;

static void print_buffer_init(printbuffer *b, char *buf, size_t len, int fmt)
{
    b->buffer = buf;
    b->length = len;
    b->offset = 0;
    b->depth = 0;
    b->format = fmt;
}

static int print_string_ptr(const char * const str, printbuffer * const p)
{
    size_t len;
    if (!str) return 1;
    len = strlen(str);
    if (p->offset + len + 3 >= p->length) return 0;
    p->buffer[p->offset++] = '\"';
    memcpy(p->buffer + p->offset, str, len);
    p->offset += len;
    p->buffer[p->offset++] = '\"';
    p->buffer[p->offset] = '\0';
    return 1;
}

static int print_value(const cJSON * const item, printbuffer * const p);

static int print_number(const cJSON * const item, printbuffer * const p)
{
    char str[64];
    int len;
    if (item->valuedouble == (double)item->valueint)
    {
        len = sprintf(str, "%d", item->valueint);
    }
    else
    {
        len = sprintf(str, "%.6g", item->valuedouble);
    }
    if (p->offset + len + 1 >= p->length) return 0;
    memcpy(p->buffer + p->offset, str, len);
    p->offset += len;
    p->buffer[p->offset] = '\0';
    return 1;
}

static int print_array(const cJSON * const item, printbuffer * const p)
{
    cJSON *child = item->child;
    if (p->offset + 2 >= p->length) return 0;
    p->buffer[p->offset++] = '[';
    while (child)
    {
        if (!print_value(child, p)) return 0;
        if (child->next)
        {
            if (p->offset + 2 >= p->length) return 0;
            p->buffer[p->offset++] = ',';
            if (p->format) p->buffer[p->offset++] = ' ';
        }
        child = child->next;
    }
    if (p->offset + 2 >= p->length) return 0;
    p->buffer[p->offset++] = ']';
    p->buffer[p->offset] = '\0';
    return 1;
}

static int print_object(const cJSON * const item, printbuffer * const p)
{
    cJSON *child = item->child;
    if (p->offset + 2 >= p->length) return 0;
    p->buffer[p->offset++] = '{';
    while (child)
    {
        if (p->format)
        {
            /* indented if needed */
        }
        if (!print_string_ptr(child->string, p)) return 0;
        if (p->offset + 2 >= p->length) return 0;
        p->buffer[p->offset++] = ':';
        if (p->format) p->buffer[p->offset++] = ' ';
        if (!print_value(child, p)) return 0;
        if (child->next)
        {
            if (p->offset + 2 >= p->length) return 0;
            p->buffer[p->offset++] = ',';
            if (p->format) p->buffer[p->offset++] = ' ';
        }
        child = child->next;
    }
    if (p->offset + 2 >= p->length) return 0;
    p->buffer[p->offset++] = '}';
    p->buffer[p->offset] = '\0';
    return 1;
}

static int print_value(const cJSON * const item, printbuffer * const p)
{
    if (!item) return 0;
    switch (item->type & 0xFF)
    {
        case cJSON_NULL:
            if (p->offset + 5 >= p->length) return 0;
            memcpy(p->buffer + p->offset, "null", 4);
            p->offset += 4;
            p->buffer[p->offset] = '\0';
            return 1;
        case cJSON_False:
            if (p->offset + 6 >= p->length) return 0;
            memcpy(p->buffer + p->offset, "false", 5);
            p->offset += 5;
            p->buffer[p->offset] = '\0';
            return 1;
        case cJSON_True:
            if (p->offset + 5 >= p->length) return 0;
            memcpy(p->buffer + p->offset, "true", 4);
            p->offset += 4;
            p->buffer[p->offset] = '\0';
            return 1;
        case cJSON_Number:
            return print_number(item, p);
        case cJSON_String:
            return print_string_ptr(item->valuestring, p);
        case cJSON_Array:
            return print_array(item, p);
        case cJSON_Object:
            return print_object(item, p);
        default:
            return 0;
    }
}

char *cJSON_Print(const cJSON *item)
{
    size_t size = 4096;
    char *buffer = (char*)malloc(size);
    printbuffer p;
    if (!buffer) return NULL;
    print_buffer_init(&p, buffer, size, 1);
    if (!print_value(item, &p))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

char *cJSON_PrintUnformatted(const cJSON *item)
{
    size_t size = 4096;
    char *buffer = (char*)malloc(size);
    printbuffer p;
    if (!buffer) return NULL;
    print_buffer_init(&p, buffer, size, 0);
    if (!print_value(item, &p))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

cJSON *cJSON_CreateNull(void) { cJSON *i = cJSON_New_Item(&global_hooks); if (i) i->type = cJSON_NULL; return i; }
cJSON *cJSON_CreateTrue(void) { cJSON *i = cJSON_New_Item(&global_hooks); if (i) { i->type = cJSON_True; i->valueint = 1; } return i; }
cJSON *cJSON_CreateFalse(void) { cJSON *i = cJSON_New_Item(&global_hooks); if (i) { i->type = cJSON_False; i->valueint = 0; } return i; }
cJSON *cJSON_CreateBool(int b) { return b ? cJSON_CreateTrue() : cJSON_CreateFalse(); }
cJSON *cJSON_CreateNumber(double num) { cJSON *i = cJSON_New_Item(&global_hooks); if (i) { i->type = cJSON_Number; i->valuedouble = num; i->valueint = (int)num; } return i; }
cJSON *cJSON_CreateString(const char *str) { cJSON *i = cJSON_New_Item(&global_hooks); if (i) { i->type = cJSON_String; i->valuestring = (char*)cJSON_strdup((const unsigned char*)str, &global_hooks); } return i; }
cJSON *cJSON_CreateArray(void) { cJSON *i = cJSON_New_Item(&global_hooks); if (i) i->type = cJSON_Array; return i; }
cJSON *cJSON_CreateObject(void) { cJSON *i = cJSON_New_Item(&global_hooks); if (i) i->type = cJSON_Object; return i; }

int cJSON_GetArraySize(const cJSON *array)
{
    cJSON *child = NULL;
    size_t size = 0;
    if (array == NULL) return 0;
    child = array->child;
    while (child != NULL) { size++; child = child->next; }
    return (int)size;
}

cJSON *cJSON_GetArrayItem(const cJSON *array, int index)
{
    cJSON *current = NULL;
    if (array == NULL || index < 0) return NULL;
    current = array->child;
    while ((current != NULL) && (index > 0)) { index--; current = current->next; }
    return current;
}

cJSON *cJSON_GetObjectItemCaseSensitive(const cJSON * const object, const char * const string)
{
    cJSON *current = NULL;
    if (object == NULL || string == NULL) return NULL;
    current = object->child;
    while (current != NULL)
    {
        if (current->string != NULL && strcmp(string, current->string) == 0) return current;
        current = current->next;
    }
    return NULL;
}

cJSON *cJSON_GetObjectItem(const cJSON * const object, const char * const string)
{
    return cJSON_GetObjectItemCaseSensitive(object, string);
}

int cJSON_HasObjectItem(const cJSON *object, const char *string)
{
    return cJSON_GetObjectItem(object, string) != NULL;
}

int cJSON_AddItemToArray(cJSON *array, cJSON *item)
{
    cJSON *child = NULL;
    if (item == NULL || array == NULL) return 0;
    child = array->child;
    if (child == NULL) { array->child = item; }
    else
    {
        while (child->next) child = child->next;
        child->next = item;
        item->prev = child;
    }
    return 1;
}

int cJSON_AddItemToObject(cJSON *object, const char *string, cJSON *item)
{
    if (item == NULL || object == NULL || string == NULL) return 0;
    if (item->string) global_hooks.deallocate(item->string);
    item->string = (char*)cJSON_strdup((const unsigned char*)string, &global_hooks);
    return cJSON_AddItemToArray(object, item);
}

cJSON *cJSON_AddNullToObject(cJSON * const object, const char * const name) { cJSON *n = cJSON_CreateNull(); cJSON_AddItemToObject(object, name, n); return n; }
cJSON *cJSON_AddTrueToObject(cJSON * const object, const char * const name) { cJSON *n = cJSON_CreateTrue(); cJSON_AddItemToObject(object, name, n); return n; }
cJSON *cJSON_AddFalseToObject(cJSON * const object, const char * const name) { cJSON *n = cJSON_CreateFalse(); cJSON_AddItemToObject(object, name, n); return n; }
cJSON *cJSON_AddBoolToObject(cJSON * const object, const char * const name, const int boolean) { return boolean ? cJSON_AddTrueToObject(object, name) : cJSON_AddFalseToObject(object, name); }
cJSON *cJSON_AddNumberToObject(cJSON * const object, const char * const name, const double number) { cJSON *n = cJSON_CreateNumber(number); cJSON_AddItemToObject(object, name, n); return n; }
cJSON *cJSON_AddStringToObject(cJSON * const object, const char * const name, const char * const string) { cJSON *n = cJSON_CreateString(string); cJSON_AddItemToObject(object, name, n); return n; }
