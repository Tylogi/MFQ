#define UNICODE
#define _UNICODE
#include <windows.h>
#include <string>

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    wchar_t destination[32768], launcher[32768], executable[32768], directory[32768];
    if (!GetEnvironmentVariableW(L"MFQ_TEST_LAUNCH_LOG", destination, 32768)) return 1;
    if (!GetEnvironmentVariableW(L"MFQ_STUDIO_LAUNCHER", launcher, 32768)) return 2;
    GetModuleFileNameW(nullptr, executable, 32768);
    GetCurrentDirectoryW(32768, directory);
    const std::wstring text = std::wstring(launcher) + L"\n" + executable + L"\n" + directory;
    HANDLE file = CreateFileW(destination, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return 3;
    DWORD written;
    const BOOL success = WriteFile(file, text.data(), static_cast<DWORD>(text.size() * sizeof(wchar_t)), &written, nullptr);
    CloseHandle(file);
    return success ? 0 : 4;
}
