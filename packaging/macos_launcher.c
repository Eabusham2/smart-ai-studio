/* Keep the Tk process inside SmartAI.app instead of launching a bare Python.
 * Built on macOS against the headers shipped with our pinned Python runtime.
 * CPython is loaded from the runtime selected by bootstrap_runtime.py; neither
 * a system Python nor a user-installed compiler is needed at launch time.
 */
#include <Python.h>
#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef SMARTAI_PYTHON_LIBRARY
#error SMARTAI_PYTHON_LIBRARY must name the packaged CPython shared library
#endif

static int fail(const char *message) {
    fprintf(stderr, "Smart AI Studio could not start: %s\n", message);
    return 1;
}

static int joined(char *destination, size_t size, const char *root, const char *suffix) {
    int count = snprintf(destination, size, "%s/%s", root, suffix);
    return count >= 0 && (size_t)count < size;
}

static int bundle_string(CFBundleRef bundle, CFStringRef key, char *value, size_t size) {
    CFTypeRef item = CFBundleGetValueForInfoDictionaryKey(bundle, key);
    return item && CFGetTypeID(item) == CFStringGetTypeID()
        && CFStringGetCString((CFStringRef)item, value, (CFIndex)size, kCFStringEncodingUTF8);
}

int main(int argc, char **argv) {
    char bundle_path[PATH_MAX], code[PATH_MAX], bundled_runtime[PATH_MAX];
    char bootstrap[PATH_MAX], library[PATH_MAX], python[PATH_MAX];
    char name[256], identifier[256];
    CFBundleRef bundle = CFBundleGetMainBundle();
    if (!bundle) return fail("The application bundle is unavailable. Re-extract the complete app.");
    CFURLRef bundle_url = CFBundleCopyBundleURL(bundle);
    int found = bundle_url && CFURLGetFileSystemRepresentation(
        bundle_url, true, (UInt8 *)bundle_path, sizeof(bundle_path));
    if (bundle_url) CFRelease(bundle_url);
    if (!found || !bundle_string(bundle, CFSTR("CFBundleName"), name, sizeof(name))
        || !bundle_string(bundle, CFSTR("CFBundleIdentifier"), identifier, sizeof(identifier))) {
        return fail("The application bundle identity is missing.");
    }
    if (!joined(code, sizeof(code), bundle_path, "Contents/Resources/app")
        || !joined(bundled_runtime, sizeof(bundled_runtime), code, "runtime/python")
        || !joined(bootstrap, sizeof(bootstrap), code, "bootstrap_runtime.py")) {
        return fail("The application path is too long.");
    }

    const char *runtime = bundled_runtime;
    const char *script = bootstrap;
    int first_argument = 1;
    /* This handoff is also used after Update & Restart and for a Setup runtime
     * stored outside the .app. The executable itself stays inside the bundle.
     */
    if (argc > 1 && strcmp(argv[1], "--smartai-run-python") == 0) {
        if (argc < 4 || argv[2][0] != '/' || argv[3][0] != '/') {
            return fail("Invalid private-runtime launch arguments.");
        }
        runtime = argv[2];
        script = argv[3];
        first_argument = 4;
    }
    if (!joined(library, sizeof(library), runtime, "lib/" SMARTAI_PYTHON_LIBRARY)
        || !joined(python, sizeof(python), runtime, "bin/python3")) {
        return fail("The private runtime path is too long.");
    }
    if (access(python, X_OK) || access(script, R_OK) || chdir(code)) {
        return fail("The packaged runtime or application script is missing or inaccessible.");
    }
    void *handle = dlopen(library, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) return fail(dlerror());

    typedef void (*init_config_fn)(PyConfig *);
    typedef PyStatus (*set_string_fn)(PyConfig *, wchar_t **, const char *);
    typedef PyStatus (*set_argv_fn)(PyConfig *, Py_ssize_t, char *const *);
    typedef PyStatus (*initialize_fn)(const PyConfig *);
    typedef void (*clear_config_fn)(PyConfig *);
    typedef int (*status_fn)(PyStatus);
    typedef int (*run_main_fn)(void);
#define LOAD(type, local, symbol) \
    type local = (type)dlsym(handle, symbol); \
    if (!local) return fail("The packaged Python does not provide " symbol ".")
    LOAD(init_config_fn, init_config, "PyConfig_InitPythonConfig");
    LOAD(set_string_fn, set_string, "PyConfig_SetBytesString");
    LOAD(set_argv_fn, set_argv, "PyConfig_SetBytesArgv");
    LOAD(initialize_fn, initialize, "Py_InitializeFromConfig");
    LOAD(clear_config_fn, clear_config, "PyConfig_Clear");
    LOAD(status_fn, status_exception, "PyStatus_Exception");
    LOAD(status_fn, status_exit, "PyStatus_IsExit");
    LOAD(run_main_fn, run_main, "Py_RunMain");
#undef LOAD

    /* Native package smoke reads these values from the actual main bundle. */
    if (setenv("SMARTAI_MACOS_NATIVE_LAUNCH", "1", 1)
        || setenv("SMARTAI_MACOS_APP_BUNDLE", bundle_path, 1)
        || setenv("SMARTAI_MACOS_APP_NAME", name, 1)
        || setenv("SMARTAI_MACOS_APP_IDENTIFIER", identifier, 1)) {
        return fail("Could not initialize the application environment.");
    }
    unsetenv("PYTHONHOME");
    unsetenv("PYTHONPATH");
    unsetenv("PYTHONUSERBASE");
    char **python_argv = calloc((size_t)argc + 3, sizeof(char *));
    if (!python_argv) return fail("Not enough memory to initialize Python.");
    int python_argc = 2;
    python_argv[0] = python;
    python_argv[1] = (char *)script;
    for (int index = first_argument; index < argc; index++) {
        /* Older LaunchServices versions can append a process serial number. */
        if (strncmp(argv[index], "-psn_", 5) != 0) python_argv[python_argc++] = argv[index];
    }

    PyConfig config;
    init_config(&config);
    config.use_environment = 0;
    config.user_site_directory = 0;
    config.parse_argv = 1;
    config.safe_path = 0; /* The selected application script owns its import directory. */
    PyStatus status = set_string(&config, &config.home, runtime);
#define CONFIGURE(expression) do { \
    if (status_exception(status)) goto initialization_failed; \
    status = (expression); \
} while (0)
    CONFIGURE(set_string(&config, &config.program_name, python));
    CONFIGURE(set_string(&config, &config.executable, python));
    CONFIGURE(set_string(&config, &config.base_executable, python));
    CONFIGURE(set_argv(&config, python_argc, python_argv));
    CONFIGURE(initialize(&config));
#undef CONFIGURE
    if (status_exception(status)) goto initialization_failed;
    clear_config(&config);
    free(python_argv);
    return run_main();

initialization_failed:
    if (!status_exit(status)) fail(status.err_msg ? status.err_msg : "Python initialization failed.");
    int result = status_exit(status) ? status.exitcode : 1;
    clear_config(&config);
    free(python_argv);
    return result;
}
