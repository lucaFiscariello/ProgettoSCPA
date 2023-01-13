#ifndef LOGGER_H_INCLUDED
#define LOGGER_H_INCLUDED

#define MSG_MAX_LEN 1024
#ifndef LOG_LEVEL
#define LOG_LEVEL 1  // Only messages with a Tag <= LOG_LEVEL will be displayed
#endif

/**
 * Interprets the non-null return value of a function as an error code and logs the
 * specified message
*/
#define ON_ERROR_LOG(ret, msg, ...) { \
    if (ret != 0){  \
        logMsg(E, msg, __VA_ARGS__); \
    } \
} \

/**
 * Interprets the non-null return value of a function as an error code and logs the
 * specified message, then returns the error code to the caller of the function.
*/
#define ON_ERROR_LOG_AND_RETURN(ret, msg, ...) { \
    if (ret != 0){  \
        logMsg(E, msg, __VA_ARGS__); \
        return ret; \
    } \
}

enum Tag
{

    I, // Informative messages
    E, // Errors
    W, // Warnings
    D  // debug messages. Log level should shall be < D when the project is finished

};

void logMsg(enum Tag tag, const char* msg, ...);

#endif // LOGGER_H_INCLUDED