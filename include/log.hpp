#pragma once
#include <array>
#include <cstdio>
#include <map>
#include <string>
#include <stdarg.h>
#include <string_view>
#ifndef MD_LOG_LEVEL
  #define MD_LOG_LEVEL 15
#endif

namespace md {
  // An enum class with the different log levels
  enum level {
    CRITICAL = 0,
    ERROR,
    EXCEPTION,
    WARNING,
    MESSAGE,
    STDERR,
    STDOUT,
    DEBUG,
    DEBUG1,
    DEBUG2,
    DEBUG3,
    DEBUG4,
    DEBUG5,
    DEBUG6,
    DEBUG7
  };

  constexpr std::array<const char*, 15> levelNames = {
      "CRITICAL", "ERROR",  "EXCEPTION", "WARNING", "MESSAGE",
      "STDERR",   "STDOUT", "DEBUG",     "DEBUG1",  "DEBUG2",
      "DEBUG3",   "DEBUG4", "DEBUG5",    "DEBUG6",  "DEBUG7"};

  constexpr std::array<const char*, 15> levelColors = {
      "\e[101m", "\e[91m", "\e[1m\e[91m", "\e[93m", "\e[92m",
      "\e[0m",   "\e[0m",  "\e[96m",      "\e[96m", "\e[96m",
      "\e[96m",  "\e[96m", "\e[96m",      "\e[96m", "\e[96m"};

  namespace detail {
    template <int level> inline auto getLogLevelDecorator() {
#ifdef MD_LOG_NO_COLORS
      constepxr auto color = "";
#else
      const std::string color(levelColors[level]);
#endif
      auto decorator = color + "[" + std::string(levelNames[level]) + "]";
      return decorator;
    }
  } // namespace detail

  template <int level> static inline auto getLogLevelInfo() {
#ifdef MD_LOG_NO_COLORS
    const std::string colorDefault = "";
#else
    const std::string colorDefault = "\e[0m";
#endif
    auto stream = stderr;
    if constexpr (level == STDOUT)
      stream = stdout;
    auto decorator = detail::getLogLevelDecorator<level>();
    std::string endline = "\n";
    if constexpr (level == CRITICAL) {
      endline = colorDefault + "\n";
    } else {
      decorator += colorDefault;
    }
    return std::make_tuple(stream, decorator, endline);
  }

  /**
   * @brief A function to log a message to the console
   * @param level The log level of the message
   * @param fmt The format string
   * @param ... The arguments to the format string
   */
  template <int level = MESSAGE> static inline void log(char const* fmt, ...) {
    if constexpr (level <= MD_LOG_LEVEL) {
      const auto currentLevelInfo = getLogLevelInfo<level>();
      const auto stream = std::get<0>(currentLevelInfo);
      const auto prefix = std::get<1>(currentLevelInfo);
      const auto suffix = std::get<2>(currentLevelInfo);
      va_list args;
      va_start(args, fmt);
      fprintf(stream, "%s ", prefix.c_str());
      vfprintf(stream, fmt, args);
      fprintf(stream, "%s", suffix.c_str());
      va_end(args);
    }
  }

  /**
   * @brief A function to log a message to the console
   * @param level The log level of the message
   * @param msg The message to log
   */
  template <int level = MESSAGE>
  static inline void log(const std::string& msg) {
    log<level>("%s", msg.c_str());
  }

} // namespace md
