#ifndef LOGGER_H_INCLUDED
#define LOGGER_H_INCLUDED

#define MSG_MAX_LEN 1024
#ifndef LOG_LEVEL
#define LOG_LEVEL 1  // Only messages with a Tag <= LOG_LEVEL will be displayed
#endif

enum Tag {

    I,          // Informative messages
    E,          // Errors
    W,          // Warnings
    D           // debug messages. Log level should shall be < D when the project is finished

};

void logMsg(enum Tag tag, const char* msg, ...);

#endif // LOGGER_H_INCLUDED