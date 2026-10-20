#pragma once

#include <chrono>
#include <cstdlib>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace server_test {

inline std::string read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), {}};
}

class Process {
public:
    Process(const std::filesystem::path& executable, const std::vector<std::string>& arguments,
            std::filesystem::path log, std::map<std::string, std::string> environment = {}) : log_(std::move(log)) {
        environment.emplace("MFQ_SERVER_API_KEY", "");
        environment.emplace("MFQ_SERVER_MODEL_DIRS", "");
        environment["PATH"] = "";
#ifdef _WIN32
        const auto quote = [](const std::wstring& argument) {
            std::wstring result = L"\"";
            std::size_t slashes = 0;
            for (const auto c : argument) {
                if (c == L'\\') { ++slashes; continue; }
                result.append(c == L'"' ? 2 * slashes + 1 : slashes, L'\\');
                result += c;
                slashes = 0;
            }
            result.append(2 * slashes, L'\\');
            return result + L'"';
        };
        auto command = quote(executable.native());
        for (const auto& argument : arguments) command += L" " + quote(std::filesystem::u8path(argument).native());
        struct Less {
            bool operator()(const std::wstring& a, const std::wstring& b) const { return _wcsicmp(a.c_str(), b.c_str()) < 0; }
        };
        std::map<std::wstring, std::wstring, Less> variables;
        auto* inherited = GetEnvironmentStringsW();
        if (!inherited) throw std::runtime_error("GetEnvironmentStrings failed");
        for (const auto* current = inherited; *current; current += std::wcslen(current) + 1) {
            const std::wstring entry(current);
            const auto equal = entry.find(L'=', entry.front() == L'=' ? 1 : 0);
            if (equal != std::wstring::npos) variables[entry.substr(0, equal)] = entry.substr(equal + 1);
        }
        FreeEnvironmentStringsW(inherited);
        for (const auto& variable : environment)
            variables[std::filesystem::u8path(variable.first).native()] = std::filesystem::u8path(variable.second).native();
        std::wstring block;
        for (const auto& variable : variables) block += variable.first + L'=' + variable.second + L'\0';
        block += L'\0';
        SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        HANDLE input = nullptr;
        if (!CreatePipe(&input, &input_, &security, 0)) throw std::runtime_error("CreatePipe failed");
        SetHandleInformation(input_, HANDLE_FLAG_INHERIT, 0);
        const auto output = CreateFileW(log_.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
            &security, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = input;
        startup.hStdOutput = startup.hStdError = output;
        PROCESS_INFORMATION child{};
        const auto started = output != INVALID_HANDLE_VALUE && CreateProcessW(executable.c_str(), command.data(),
            nullptr, nullptr, TRUE, CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW, block.data(), nullptr, &startup, &child);
        CloseHandle(input);
        if (output != INVALID_HANDLE_VALUE) CloseHandle(output);
        if (!started) { CloseHandle(input_); input_ = nullptr; throw std::runtime_error("CreateProcess failed"); }
        process_ = child.hProcess;
        pid_ = child.dwProcessId;
        CloseHandle(child.hThread);
#else
        std::vector<std::string> values{executable.string()};
        values.insert(values.end(), arguments.begin(), arguments.end());
        std::vector<char*> argv;
        for (auto& value : values) argv.push_back(value.data());
        argv.push_back(nullptr);
        int input[2];
        if (::pipe(input)) throw std::runtime_error("pipe failed");
        const int output = ::open(log_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (output < 0) { ::close(input[0]); ::close(input[1]); throw std::runtime_error("cannot open child log"); }
        pid_ = ::fork();
        if (pid_ == 0) {
            ::dup2(input[0], STDIN_FILENO);
            ::dup2(output, STDOUT_FILENO);
            ::dup2(output, STDERR_FILENO);
            ::close(input[0]); ::close(input[1]); ::close(output);
            for (const auto& variable : environment) ::setenv(variable.first.c_str(), variable.second.c_str(), 1);
            ::execv(executable.c_str(), argv.data());
            ::_exit(127);
        }
        ::close(input[0]);
        ::close(output);
        if (pid_ < 0) { ::close(input[1]); throw std::runtime_error("fork failed"); }
        input_ = input[1];
        ::fcntl(input_, F_SETFD, FD_CLOEXEC);
#endif
    }

    ~Process() {
        close_input();
        if (running()) {
#ifdef _WIN32
            TerminateProcess(process_, 1);
#else
            ::kill(pid_, SIGKILL);
#endif
            try { wait(); } catch (...) {}
        }
#ifdef _WIN32
        if (process_) CloseHandle(process_);
#endif
    }
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;

    bool running() {
        if (exited_) return false;
#ifdef _WIN32
        if (WaitForSingleObject(process_, 0) == WAIT_TIMEOUT) return true;
        DWORD code = 0;
        GetExitCodeProcess(process_, &code);
        result_ = static_cast<int>(code);
#else
        int status = 0;
        const auto result = ::waitpid(pid_, &status, WNOHANG);
        if (result == 0 || (result < 0 && errno == EINTR)) return true;
        if (result < 0) return false;
        result_ = WIFSIGNALED(status) ? -WTERMSIG(status) : WEXITSTATUS(status);
#endif
        exited_ = true;
        return false;
    }
    int wait() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
        while (running()) {
            if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("child process did not exit: " + output());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return result_;
    }
    void close_input() {
#ifdef _WIN32
        if (input_) { CloseHandle(input_); input_ = nullptr; }
#else
        if (input_ >= 0) { ::close(input_); input_ = -1; }
#endif
    }
    void write_input(const std::string& value) {
#ifdef _WIN32
        DWORD written = 0;
        if (!WriteFile(input_, value.data(), static_cast<DWORD>(value.size()), &written, nullptr) || written != value.size())
            throw std::runtime_error("cannot write child stdin");
#else
        if (::write(input_, value.data(), value.size()) != static_cast<ssize_t>(value.size()))
            throw std::runtime_error("cannot write child stdin");
#endif
    }
    std::string output() const { return read_file(log_); }
    unsigned long pid() const { return static_cast<unsigned long>(pid_); }
#ifndef _WIN32
    void signal(int value) { if (::kill(pid_, value)) throw std::runtime_error("kill failed"); }
#endif

private:
    std::filesystem::path log_;
    bool exited_ = false;
    int result_ = 0;
#ifdef _WIN32
    HANDLE process_ = nullptr, input_ = nullptr;
    DWORD pid_ = 0;
#else
    pid_t pid_ = -1;
    int input_ = -1;
#endif
};

} // namespace server_test
