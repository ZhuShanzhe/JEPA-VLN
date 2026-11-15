/* Select a Mesa software EGL device for a dedicated headless Habitat process.
 * Other processes and the system EGL configuration are unaffected. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

typedef void *Device;
typedef void (*Proc)(void);
typedef Proc (*Resolver)(const char *);
typedef unsigned (*ListDevices)(int, Device *, int *);
typedef const char *(*QueryString)(Device, int);

static Resolver resolver(void) {
    Resolver fn = (Resolver)dlsym(RTLD_NEXT, "eglGetProcAddress");
    if (!fn) {
        void *egl = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
        if (egl) fn = (Resolver)dlsym(egl, "eglGetProcAddress");
    }
    if (!fn) abort();
    return fn;
}

static unsigned list_software(int capacity, Device *out, int *count) {
    ListDevices list = (ListDevices)resolver()("eglQueryDevicesEXT");
    QueryString query = (QueryString)resolver()("eglQueryDeviceStringEXT");
    Device devices[32];
    int total = 0;
    if (!list || !query || !count || !list(32, devices, &total)) return 0;
    *count = 0;
    for (int i = 0; i < total; ++i) {
        const char *extensions = query(devices[i], 0x3055); /* EGL_EXTENSIONS */
        if (extensions && strstr(extensions, "EGL_MESA_device_software")) {
            if (out && capacity > 0) out[0] = devices[i];
            *count = 1;
            return 1;
        }
    }
    return 1;
}

Proc eglGetProcAddress(const char *name) {
    if (!strcmp(name, "eglQueryDevicesEXT")) return (Proc)list_software;
    return resolver()(name);
}
