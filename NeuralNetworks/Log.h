#pragma once

#include <string>

void platformLogMessage(int level, const char* message);

void waitHere();

// to_string

static char to_string_buffer[1024 * 10];

template <typename T>
static const char* to_string(const T& t)
{
	sprintf_s(to_string_buffer, "Container: %d", t);
	return to_string_buffer;
}

template <>
static const char* to_string<char>(const char& t)
{
	return &t;
}

template <>
static const char* to_string<char*>(char* const& t)
{
	return t;
}

template <>
static const char* to_string<unsigned char const*>(unsigned char const* const& t)
{
	return (const char*)t;
}

template <>
static const char* to_string<const char*>(const char* const& t)
{
	return t;
}

template <>
static const char* to_string<std::string>(const std::string& t)
{
	return t.c_str();
}

template <>
static const char* to_string<bool>(const bool& t)
{
	return t ? "true" : "false";
}

// stuff

template <typename First>
static void print_log_internal(char* buffer, int& position, First&& first)
{
	const char* formatted = to_string<First>(first);
	int length = strlen(formatted);
	memcpy(&buffer[position], formatted, length);
	position += length;
}

template <typename First, typename... Args>
static void print_log_internal(char* buffer, int& position, First&& first, Args&&... args)
{
	const char* formatted = to_string<First>(first);
	int length = strlen(formatted);
	memcpy(&buffer[position], formatted, length);
	position += length;
	if (sizeof...(Args))
		print_log_internal(buffer, position, std::forward<Args>(args)...);
}

template <typename... Args>
static void log_message(int level, bool newline, Args... args)
{
	char buffer[1024 * 10];
	int position = 0;
	print_log_internal(buffer, position, std::forward<Args>(args)...);

	if (newline)
		buffer[position++] = '\n';

	buffer[position] = 0;

	platformLogMessage(level, buffer);
}

#define LOG_INFO(...) log_message(0, true, "INFO:    ", __VA_ARGS__)
#define LOG_WARN(...) log_message(1, true, "WARN:    ", __VA_ARGS__)
#define LOG_ERROR(...) log_message(2, true, "ERROR:    ", __VA_ARGS__)
#define LOG_FATAL(...) log_message(3, true, "FATAL:    ", __VA_ARGS__)