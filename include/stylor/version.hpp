#ifndef STYLOR_VERSION_HPP
#define STYLOR_VERSION_HPP

#define STYLOR_VERSION_MAJOR 0
#define STYLOR_VERSION_MINOR 1
#define STYLOR_VERSION_PATCH 0

namespace stylor {
    /// @brief Major version.
    inline constexpr int version_major = STYLOR_VERSION_MAJOR;
    /// @brief Minor version.
    inline constexpr int version_minor = STYLOR_VERSION_MINOR;
    /// @brief Patch version.
    inline constexpr int version_patch = STYLOR_VERSION_PATCH;
    /// @brief String representation of library version (observes semantic versioning convention).
    inline constexpr char version_string[] = {
        '0' + version_major, '.', '0' + version_minor, '.', '0' + version_patch, '\0'
    };
} // namespace stylor

#endif // STYLOR_VERSION_HPP
