#ifndef CONFIG_MANAGER_HPP
#define CONFIG_MANAGER_HPP

#include <string>
#include <map>
#include <memory>

namespace WaferVision {

class ConfigManager {
public:
    static ConfigManager& getInstance();
    
    void loadFromFile(const std::string& path);
    void saveToFile(const std::string& path);
    
    void set(const std::string& key, const std::string& value);
    void set(const std::string& key, int value);
    void set(const std::string& key, double value);
    void set(const std::string& key, bool value);
    
    std::string getString(const std::string& key, const std::string& defaultValue = "");
    int getInt(const std::string& key, int defaultValue = 0);
    double getDouble(const std::string& key, double defaultValue = 0.0);
    bool getBool(const std::string& key, bool defaultValue = false);
    
    bool has(const std::string& key) const;
    void remove(const std::string& key);
    void clear();
    
private:
    ConfigManager() = default;
    ~ConfigManager() = default;
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
    
    std::map<std::string, std::string> m_config;
};

} // namespace WaferVision

#endif // CONFIG_MANAGER_HPP
