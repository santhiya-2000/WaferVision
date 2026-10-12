#include "ConfigManager.hpp"
#include <fstream>
#include <sstream>

namespace WaferVision {

ConfigManager& ConfigManager::getInstance() {
    static ConfigManager instance;
    return instance;
}

void ConfigManager::loadFromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Parse key=value
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            m_config[key] = value;
        }
    }
    
    file.close();
}

void ConfigManager::saveToFile(const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        return;
    }
    
    file << "# WaferVision Configuration File\n";
    file << "# Auto-generated\n\n";
    
    for (const auto& pair : m_config) {
        file << pair.first << " = " << pair.second << "\n";
    }
    
    file.close();
}

void ConfigManager::set(const std::string& key, const std::string& value) {
    m_config[key] = value;
}

void ConfigManager::set(const std::string& key, int value) {
    m_config[key] = std::to_string(value);
}

void ConfigManager::set(const std::string& key, double value) {
    m_config[key] = std::to_string(value);
}

void ConfigManager::set(const std::string& key, bool value) {
    m_config[key] = value ? "true" : "false";
}

std::string ConfigManager::getString(const std::string& key, const std::string& defaultValue) {
    auto it = m_config.find(key);
    if (it != m_config.end()) {
        return it->second;
    }
    return defaultValue;
}

int ConfigManager::getInt(const std::string& key, int defaultValue) {
    auto it = m_config.find(key);
    if (it != m_config.end()) {
        try {
            return std::stoi(it->second);
        } catch (...) {
            return defaultValue;
        }
    }
    return defaultValue;
}

double ConfigManager::getDouble(const std::string& key, double defaultValue) {
    auto it = m_config.find(key);
    if (it != m_config.end()) {
        try {
            return std::stod(it->second);
        } catch (...) {
            return defaultValue;
        }
    }
    return defaultValue;
}

bool ConfigManager::getBool(const std::string& key, bool defaultValue) {
    auto it = m_config.find(key);
    if (it != m_config.end()) {
        std::string value = it->second;
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        return value == "true" || value == "1" || value == "yes";
    }
    return defaultValue;
}

bool ConfigManager::has(const std::string& key) const {
    return m_config.find(key) != m_config.end();
}

void ConfigManager::remove(const std::string& key) {
    m_config.erase(key);
}

void ConfigManager::clear() {
    m_config.clear();
}

} // namespace WaferVision
