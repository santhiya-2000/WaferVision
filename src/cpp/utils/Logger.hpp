#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <string>
#include <fstream>
#include <mutex>
#include <memory>
#include <sstream>

namespace WaferVision {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

class Logger {
public:
    static Logger& getInstance();
    
    void setLogFile(const std::string& path);
    void setLogLevel(LogLevel level);
    void setConsoleOutput(bool enabled);
    
    void log(LogLevel level, const std::string& message);
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);
    void critical(const std::string& message);
    
private:
    Logger();
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    std::string formatMessage(LogLevel level, const std::string& message);
    std::string levelToString(LogLevel level);
    
    std::ofstream m_logFile;
    LogLevel m_logLevel;
    bool m_consoleOutput;
    std::mutex m_mutex;
};

// Convenience macros
#define LOG_DEBUG(msg) WaferVision::Logger::getInstance().debug(msg)
#define LOG_INFO(msg) WaferVision::Logger::getInstance().info(msg)
#define LOG_WARNING(msg) WaferVision::Logger::getInstance().warning(msg)
#define LOG_ERROR(msg) WaferVision::Logger::getInstance().error(msg)
#define LOG_CRITICAL(msg) WaferVision::Logger::getInstance().critical(msg)

} // namespace WaferVision

#endif // LOGGER_HPP
