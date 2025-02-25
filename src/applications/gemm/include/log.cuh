#ifndef __AEOLUS_LOG_CUH
#define __AEOLUS_LOG_CUH

#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>

#define AEOLUS_LOG_DEBUG(...) AEOLUS_LOG(AEOLUS_LOG_LEVEL_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define AEOLUS_LOG_INFO(...) AEOLUS_LOG(AEOLUS_LOG_LEVEL_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define AEOLUS_LOG_WARNING(...) AEOLUS_LOG(AEOLUS_LOG_LEVEL_WARNING, __FILE__, __LINE__, __VA_ARGS__)
#define AEOLUS_LOG_ERROR(...) AEOLUS_LOG(AEOLUS_LOG_LEVEL_ERROR, __FILE__, __LINE__, __VA_ARGS__)

#ifndef __CUDA_ARCH__
    #define AEOLUS_LOG(level, filename, lineno, ...) aeolus_log(level, filename, lineno, __VA_ARGS__); 
#else
    #define AEOLUS_LOG(level, filename, lineno, ...) \
        printf("%s:%d ", filename, lineno); \
        printf(__VA_ARGS__); 
#endif

enum aeolus_log_level
{
    AEOLUS_LOG_LEVEL_NULL = 0,
    AEOLUS_LOG_LEVEL_DEBUG = 1,
    AEOLUS_LOG_LEVEL_INFO = 2,
    AEOLUS_LOG_LEVEL_WARNING = 3,
    AEOLUS_LOG_LEVEL_ERROR = 4,
};

static aeolus_log_level log_level = AEOLUS_LOG_LEVEL_NULL;

inline aeolus_log_level get_log_level()
{
    if (log_level != AEOLUS_LOG_LEVEL_NULL)
    {
        return log_level;
    }

    char *log_level_env = getenv("AEOLUS_LOG_LEVEL");
    if (log_level_env == NULL)
    {
#ifdef __DEBUG__
        log_level = AEOLUS_LOG_LEVEL_INFO;
#else
        log_level = AEOLUS_LOG_LEVEL_WARNING;
#endif
    } else
    {
        if (strcmp(log_level_env, "DEBUG") == 0)
        {
            log_level = AEOLUS_LOG_LEVEL_DEBUG;
        } else if (strcmp(log_level_env, "INFO") == 0)
        {
            log_level = AEOLUS_LOG_LEVEL_INFO;
        } else if (strcmp(log_level_env, "WARNING") == 0)
        {
            log_level = AEOLUS_LOG_LEVEL_WARNING;
        } else if (strcmp(log_level_env, "ERROR") == 0)
        {
            log_level = AEOLUS_LOG_LEVEL_ERROR;
        } else
        {
            log_level = AEOLUS_LOG_LEVEL_INFO;
        }
    }

    return log_level;
}

__host__ inline void aeolus_log(aeolus_log_level level, const char *filename, int lineno, const char *format, ...)
{
    if (level < get_log_level())
    {
        return;
    }

    char *level_str;
    switch (level)
    {
        case AEOLUS_LOG_LEVEL_DEBUG:
            level_str = (char *)"DEBUG";
            break;
        case AEOLUS_LOG_LEVEL_INFO:
            level_str = (char *)"INFO";
            break;
        case AEOLUS_LOG_LEVEL_WARNING:
            level_str = (char *)"WARNING";
            break;
        case AEOLUS_LOG_LEVEL_ERROR:
            level_str = (char *)"ERROR";
            break;
        default:
            level_str = (char *)"UNKNOWN";
            break;
    }

    va_list args;
    va_start(args, format);
    fprintf(stderr, "[%s] %s:%d ", level_str, filename, lineno);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

#endif