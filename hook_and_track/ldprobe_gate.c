// LD_PRELOAD probe for EdgeTPU/TFLite runs
// - Intercept ioctl to log USBDEVFS_{SUBMITURB,REAPURB,REAPURBNDELAY}
// - Intercept memcpy/memmove to observe userland copies around invoke
// - Timestamps use CLOCK_BOOTTIME to align with usbmon timelines

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/usbdevice_fs.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

// ---------- utils ----------
static inline double now_boottime_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_BOOTTIME, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int log_fd = -1;
static pthread_mutex_t log_mu = PTHREAD_MUTEX_INITIALIZER;
static __thread int reent = 0; // recursion guard per thread

static void safe_write(const char *s, size_t n) {
  if (log_fd < 0) return;
  ssize_t _ = write(log_fd, s, n);
  (void)_;
}

static void log_line(const char *buf) {
  size_t n = strlen(buf);
  safe_write(buf, n);
  safe_write("\n", 1);
}

static void init_log_fd_once(void) {
  if (log_fd >= 0) return;
  if (reent) return;
  reent++;
  const char *p = getenv("LDP_LOG");
  char path[512];
  if (!p || !*p) {
    snprintf(path, sizeof(path), "/tmp/ldprobe_gate_%d.jsonl", (int)getpid());
    p = path;
  }
  int fd = open(p, O_CREAT | O_WRONLY | O_APPEND, 0644);
  if (fd >= 0) {
    log_fd = fd;
  }
  reent--;
}

__attribute__((constructor)) static void gate_ctor(void) {
  init_log_fd_once();
}

// ---------- real funcs ----------
static int (*real_ioctl)(int, unsigned long, ...) = NULL;
static void *(*real_memcpy)(void *, const void *, size_t) = NULL;
static void *(*real_memmove)(void *, const void *, size_t) = NULL;
static void *(*real_memset)(void *, int, size_t) = NULL;
static void *(*real___memcpy_chk)(void *, const void *, size_t, size_t) = NULL;
static void *(*real___memmove_chk)(void *, const void *, size_t, size_t) = NULL;

static void ensure_real(void) {
  if (!real_ioctl) real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");
  if (!real_memcpy) real_memcpy = (void *(*)(void *, const void *, size_t))dlsym(RTLD_NEXT, "memcpy");
  if (!real_memmove) real_memmove = (void *(*)(void *, const void *, size_t))dlsym(RTLD_NEXT, "memmove");
  if (!real_memset) real_memset = (void *(*)(void *, int, size_t))dlsym(RTLD_NEXT, "memset");
  if (!real___memcpy_chk) real___memcpy_chk = (void *(*)(void *, const void *, size_t, size_t))dlsym(RTLD_NEXT, "__memcpy_chk");
  if (!real___memmove_chk) real___memmove_chk = (void *(*)(void *, const void *, size_t, size_t))dlsym(RTLD_NEXT, "__memmove_chk");
}

// ---------- runtime config ----------
static size_t memcpy_min_n(void) {
  const char *s = getenv("LDP_MEM_THRESHOLD");
  if (!s || !*s) return 64; // ignore tiny copies
  long v = strtol(s, NULL, 10);
  if (v < 0) v = 0;
  return (size_t)v;
}

static int capture_mem_only_in_invoke(void) {
  const char *s = getenv("LDP_MEM_ONLY_IN_INVOKE");
  if (!s || !*s) return 1; // default only when invoke bracket is active
  return (strcmp(s, "0") != 0);
}

// ---------- invoke brackets (populated by invoke.cpp) ----------
// Exposed as weak symbols so this file can compile standalone.
__attribute__((weak)) int ldprobe_get_invoke_id(void) { return 0; }
__attribute__((weak)) int ldprobe_is_invoke_active(void) { return 0; }

// ---------- tracking helpers ----------
typedef struct {
  void *urbp;            // user-space pointer to struct usbdevfs_urb
  void *buf;             // user buffer
  size_t len;            // requested length
  unsigned char ep;      // endpoint (bEndpointAddress)
  int dir_in;            // 1 if IN (device->host), 0 if OUT
  double t_submit;       // submit time
  int invoke_id;         // snapshot at submit time
} urb_track_t;

#define MAX_URB_TRACK 8192
static urb_track_t g_urbs[MAX_URB_TRACK];
static pthread_mutex_t g_urbs_mu = PTHREAD_MUTEX_INITIALIZER;

static int ep_dir_in(unsigned char ep) { return (ep & 0x80) ? 1 : 0; }

static void urbs_add(void *urbp, void *buf, size_t len, unsigned char ep, double t, int inv) {
  pthread_mutex_lock(&g_urbs_mu);
  for (int i = 0; i < MAX_URB_TRACK; ++i) {
    if (g_urbs[i].urbp == NULL) {
      g_urbs[i].urbp = urbp;
      g_urbs[i].buf = buf;
      g_urbs[i].len = len;
      g_urbs[i].ep = ep;
      g_urbs[i].dir_in = ep_dir_in(ep);
      g_urbs[i].t_submit = t;
      g_urbs[i].invoke_id = inv;
      break;
    }
  }
  pthread_mutex_unlock(&g_urbs_mu);
}

static int urbs_find_index_by_urbp(void *urbp) {
  for (int i = 0; i < MAX_URB_TRACK; ++i) if (g_urbs[i].urbp == urbp) return i;
  return -1;
}

static urb_track_t urbs_pop_by_urbp(void *urbp) {
  urb_track_t out = {0};
  pthread_mutex_lock(&g_urbs_mu);
  int idx = urbs_find_index_by_urbp(urbp);
  if (idx >= 0) {
    out = g_urbs[idx];
    memset(&g_urbs[idx], 0, sizeof(g_urbs[idx]));
  }
  pthread_mutex_unlock(&g_urbs_mu);
  return out;
}

// ---------- logging helpers ----------
static void log_submit(int fd, struct usbdevfs_urb *u, int inv, double ts) {
  char buf[512];
  const char *dir = ep_dir_in(u->endpoint) ? "IN" : "OUT";
  snprintf(buf, sizeof(buf),
           "{\"ts\":%.9f,\"ev\":\"SUBMITURB\",\"fd\":%d,\"urbp\":%p,\"buf\":%p,\"len\":%u,\"ep\":%u,\"dir\":\"%s\",\"invoke\":%d}",
           ts, fd, (void*)u, u->buffer, (unsigned)u->buffer_length, (unsigned)u->endpoint, dir, inv);
  log_line(buf);
}

static void log_reap(int fd, struct usbdevfs_urb *u, int inv, double ts, const char *kind) {
  char buf[512];
  const char *dir = ep_dir_in(u->endpoint) ? "IN" : "OUT";
  snprintf(buf, sizeof(buf),
           "{\"ts\":%.9f,\"ev\":\"%s\",\"fd\":%d,\"urbp\":%p,\"buf\":%p,\"al\":%u,\"ep\":%u,\"dir\":\"%s\",\"status\":%d,\"invoke\":%d}",
           ts, kind, fd, (void*)u, u->buffer, (unsigned)u->actual_length, (unsigned)u->endpoint, dir, u->status, inv);
  log_line(buf);
}

static void log_memcpy_like(const char *name, void *dst, const void *src, size_t n, double t0, double t1, int inv) {
  char buf[512];
  snprintf(buf, sizeof(buf),
           "{\"ts\":%.9f,\"ev\":\"%s\",\"dst\":%p,\"src\":%p,\"n\":%zu,\"dt_ms\":%.3f,\"invoke\":%d}",
           t1, name, dst, src, n, (t1 - t0) * 1000.0, inv);
  log_line(buf);
}

static void log_memset_like(const char *name, void *dst, size_t n, double t0, double t1, int inv) {
  char buf[512];
  snprintf(buf, sizeof(buf),
           "{\"ts\":%.9f,\"ev\":\"%s\",\"dst\":%p,\"n\":%zu,\"dt_ms\":%.3f,\"invoke\":%d}",
           t1, name, dst, n, (t1 - t0) * 1000.0, inv);
  log_line(buf);
}

// ---------- interposed functions ----------
int ioctl(int fd, unsigned long req, ...) {
  init_log_fd_once();
  ensure_real();
  if (!real_ioctl) {
    errno = ENOSYS;
    return -1;
  }

  va_list ap;
  va_start(ap, req);
  void *arg = va_arg(ap, void*);
  va_end(ap);

  // Fast path if not USBDEVFS*
  switch (req) {
    case USBDEVFS_SUBMITURB: {
      double t = now_boottime_s();
      struct usbdevfs_urb *u = (struct usbdevfs_urb *)arg;
      int inv = ldprobe_get_invoke_id();
      log_submit(fd, u, inv, t);
      urbs_add((void*)u, u->buffer, u->buffer_length, u->endpoint, t, inv);
      return real_ioctl(fd, req, u);
    }
    case USBDEVFS_REAPURB:
    case USBDEVFS_REAPURBNDELAY: {
      int r = real_ioctl(fd, req, arg);
      if (r == 0 && arg) {
        double t = now_boottime_s();
        void *urbp = *(void **)arg; // returned pointer
        if (urbp) {
          struct usbdevfs_urb *u = (struct usbdevfs_urb *)urbp;
          int inv = ldprobe_get_invoke_id();
          log_reap(fd, u, inv, t, req == USBDEVFS_REAPURB ? "REAPURB" : "REAPURBNDELAY");
          // pop from tracker if present
          (void)urbs_pop_by_urbp(urbp);
        }
      }
      return r;
    }
    default:
      return real_ioctl(fd, req, arg);
  }
}

void *memcpy(void *dst, const void *src, size_t n) {
  init_log_fd_once();
  ensure_real();
  if (!real_memcpy) return NULL;
  if (reent) return real_memcpy(dst, src, n);
  reent++;
  const int only_invoke = capture_mem_only_in_invoke();
  const int active = ldprobe_is_invoke_active();
  const size_t min_n = memcpy_min_n();
  double t0 = 0.0;
  if ((!only_invoke || active) && n >= min_n) t0 = now_boottime_s();
  void *ret = real_memcpy(dst, src, n);
  if ((!only_invoke || active) && n >= min_n) {
    double t1 = now_boottime_s();
    log_memcpy_like("MEMCPY", dst, src, n, t0, t1, ldprobe_get_invoke_id());
  }
  reent--;
  return ret;
}

void *memmove(void *dst, const void *src, size_t n) {
  init_log_fd_once();
  ensure_real();
  if (!real_memmove) return NULL;
  if (reent) return real_memmove(dst, src, n);
  reent++;
  const int only_invoke = capture_mem_only_in_invoke();
  const int active = ldprobe_is_invoke_active();
  const size_t min_n = memcpy_min_n();
  double t0 = 0.0;
  if ((!only_invoke || active) && n >= min_n) t0 = now_boottime_s();
  void *ret = real_memmove(dst, src, n);
  if ((!only_invoke || active) && n >= min_n) {
    double t1 = now_boottime_s();
    log_memcpy_like("MEMMOVE", dst, src, n, t0, t1, ldprobe_get_invoke_id());
  }
  reent--;
  return ret;
}

// ---------- libusb function wrappers (actual usage evidence) ----------
// We avoid including libusb headers; declare opaque pointers.
typedef struct libusb_context libusb_context;
typedef struct libusb_device_handle libusb_device_handle;
typedef struct libusb_device libusb_device;

static void log_call_simple(const char *name, double t, int invoke, const char *extra_fmt, ...) {
  char buf[512];
  int n = snprintf(buf, sizeof(buf), "{\"ts\":%.9f,\"ev\":\"%s\",\"invoke\":%d", t, name, invoke);
  va_list ap;
  if (extra_fmt && *extra_fmt) {
    va_start(ap, extra_fmt);
    char extra[256];
    vsnprintf(extra, sizeof(extra), extra_fmt, ap);
    va_end(ap);
    snprintf(buf + n, sizeof(buf) - n, ",%s}", extra);
  } else {
    snprintf(buf + n, sizeof(buf) - n, "}");
  }
  log_line(buf);
}

int libusb_bulk_transfer(libusb_device_handle *devh, unsigned char endpoint, unsigned char *data, int length, int *transferred, unsigned int timeout) {
  ensure_real();
  int (*real_fn)(libusb_device_handle*, unsigned char, unsigned char*, int, int*, unsigned int) =
      (int(*)(libusb_device_handle*, unsigned char, unsigned char*, int, int*, unsigned int))dlsym(RTLD_NEXT, "libusb_bulk_transfer");
  if (!real_fn) return -1;
  double t0 = now_boottime_s();
  int r = real_fn(devh, endpoint, data, length, transferred, timeout);
  double t1 = now_boottime_s();
  int inv = ldprobe_get_invoke_id();
  int dir_in = (endpoint & 0x80) ? 1 : 0;
  log_call_simple("LIBUSB_BULK", t1, inv, "\"ep\":%u,\"len\":%d,\"dir\":\"%s\",\"rc\":%d,\"dt_ms\":%.3f", (unsigned)endpoint, length, dir_in?"IN":"OUT", r, (t1-t0)*1000.0);
  return r;
}

int libusb_interrupt_transfer(libusb_device_handle *devh, unsigned char endpoint, unsigned char *data, int length, int *transferred, unsigned int timeout) {
  ensure_real();
  int (*real_fn)(libusb_device_handle*, unsigned char, unsigned char*, int, int*, unsigned int) =
      (int(*)(libusb_device_handle*, unsigned char, unsigned char*, int, int*, unsigned int))dlsym(RTLD_NEXT, "libusb_interrupt_transfer");
  if (!real_fn) return -1;
  double t0 = now_boottime_s();
  int r = real_fn(devh, endpoint, data, length, transferred, timeout);
  double t1 = now_boottime_s();
  int inv = ldprobe_get_invoke_id();
  int dir_in = (endpoint & 0x80) ? 1 : 0;
  log_call_simple("LIBUSB_INTR", t1, inv, "\"ep\":%u,\"len\":%d,\"dir\":\"%s\",\"rc\":%d,\"dt_ms\":%.3f", (unsigned)endpoint, length, dir_in?"IN":"OUT", r, (t1-t0)*1000.0);
  return r;
}

int libusb_control_transfer(libusb_device_handle *devh, unsigned char request_type, unsigned char request,
                            unsigned short value, unsigned short index, unsigned char *data, unsigned short length,
                            unsigned int timeout) {
  ensure_real();
  int (*real_fn)(libusb_device_handle*, unsigned char, unsigned char, unsigned short, unsigned short, unsigned char*, unsigned short, unsigned int) =
      (int(*)(libusb_device_handle*, unsigned char, unsigned char, unsigned short, unsigned short, unsigned char*, unsigned short, unsigned int))dlsym(RTLD_NEXT, "libusb_control_transfer");
  if (!real_fn) return -1;
  double t0 = now_boottime_s();
  int r = real_fn(devh, request_type, request, value, index, data, length, timeout);
  double t1 = now_boottime_s();
  int inv = ldprobe_get_invoke_id();
  log_call_simple("LIBUSB_CTRL", t1, inv, "\"reqt\":%u,\"req\":%u,\"len\":%u,\"rc\":%d,\"dt_ms\":%.3f", (unsigned)request_type, (unsigned)request, (unsigned)length, r, (t1-t0)*1000.0);
  return r;
}

int libusb_submit_transfer(void *transfer) {
  ensure_real();
  int (*real_fn)(void*) = (int(*)(void*))dlsym(RTLD_NEXT, "libusb_submit_transfer");
  if (!real_fn) return -1;
  int r = real_fn(transfer);
  log_call_simple("LIBUSB_SUBMIT", now_boottime_s(), ldprobe_get_invoke_id(), "\"xfer\":%p,\"rc\":%d", transfer, r);
  return r;
}

int libusb_cancel_transfer(void *transfer) {
  ensure_real();
  int (*real_fn)(void*) = (int(*)(void*))dlsym(RTLD_NEXT, "libusb_cancel_transfer");
  if (!real_fn) return -1;
  int r = real_fn(transfer);
  log_call_simple("LIBUSB_CANCEL", now_boottime_s(), ldprobe_get_invoke_id(), "\"xfer\":%p,\"rc\":%d", transfer, r);
  return r;
}

int libusb_open(libusb_device *dev, libusb_device_handle **handle) {
  ensure_real();
  int (*real_fn)(libusb_device*, libusb_device_handle**) = (int(*)(libusb_device*, libusb_device_handle**))dlsym(RTLD_NEXT, "libusb_open");
  if (!real_fn) return -1;
  int r = real_fn(dev, handle);
  log_call_simple("LIBUSB_OPEN", now_boottime_s(), ldprobe_get_invoke_id(), "\"rc\":%d,\"hdl\":%p", r, handle ? (void*)*handle : NULL);
  return r;
}

void libusb_close(libusb_device_handle *devh) {
  ensure_real();
  void (*real_fn)(libusb_device_handle*) = (void(*)(libusb_device_handle*))dlsym(RTLD_NEXT, "libusb_close");
  if (real_fn) real_fn(devh);
  log_call_simple("LIBUSB_CLOSE", now_boottime_s(), ldprobe_get_invoke_id(), "\"hdl\":%p", devh);
}

int libusb_claim_interface(libusb_device_handle *devh, int interface_number) {
  ensure_real();
  int (*real_fn)(libusb_device_handle*, int) = (int(*)(libusb_device_handle*, int))dlsym(RTLD_NEXT, "libusb_claim_interface");
  if (!real_fn) return -1;
  int r = real_fn(devh, interface_number);
  log_call_simple("LIBUSB_CLAIM", now_boottime_s(), ldprobe_get_invoke_id(), "\"iface\":%d,\"rc\":%d", interface_number, r);
  return r;
}

int libusb_release_interface(libusb_device_handle *devh, int interface_number) {
  ensure_real();
  int (*real_fn)(libusb_device_handle*, int) = (int(*)(libusb_device_handle*, int))dlsym(RTLD_NEXT, "libusb_release_interface");
  if (!real_fn) return -1;
  int r = real_fn(devh, interface_number);
  log_call_simple("LIBUSB_RELEASE", now_boottime_s(), ldprobe_get_invoke_id(), "\"iface\":%d,\"rc\":%d", interface_number, r);
  return r;
}

void *memset(void *dst, int c, size_t n) {
  init_log_fd_once(); ensure_real(); if (!real_memset) return NULL; if (reent) return real_memset(dst, c, n);
  reent++;
  const int only_invoke = capture_mem_only_in_invoke(); const int active = ldprobe_is_invoke_active(); const size_t min_n = memcpy_min_n();
  double t0 = 0.0; if ((!only_invoke || active) && n >= min_n) t0 = now_boottime_s();
  void *ret = real_memset(dst, c, n);
  if ((!only_invoke || active) && n >= min_n) { double t1 = now_boottime_s(); log_memset_like("MEMSET", dst, n, t0, t1, ldprobe_get_invoke_id()); }
  reent--; return ret;
}

void *__memcpy_chk(void *dst, const void *src, size_t n, size_t dstlen) {
  init_log_fd_once(); ensure_real(); if (!real___memcpy_chk) return NULL; if (reent) return real___memcpy_chk(dst, src, n, dstlen);
  reent++;
  const int only_invoke = capture_mem_only_in_invoke(); const int active = ldprobe_is_invoke_active(); const size_t min_n = memcpy_min_n();
  double t0 = 0.0; if ((!only_invoke || active) && n >= min_n) t0 = now_boottime_s();
  void *ret = real___memcpy_chk(dst, src, n, dstlen);
  if ((!only_invoke || active) && n >= min_n) { double t1 = now_boottime_s(); log_memcpy_like("MEMCPY_CHK", dst, src, n, t0, t1, ldprobe_get_invoke_id()); }
  reent--; return ret;
}

void *__memmove_chk(void *dst, const void *src, size_t n, size_t dstlen) {
  init_log_fd_once(); ensure_real(); if (!real___memmove_chk) return NULL; if (reent) return real___memmove_chk(dst, src, n, dstlen);
  reent++;
  const int only_invoke = capture_mem_only_in_invoke(); const int active = ldprobe_is_invoke_active(); const size_t min_n = memcpy_min_n();
  double t0 = 0.0; if ((!only_invoke || active) && n >= min_n) t0 = now_boottime_s();
  void *ret = real___memmove_chk(dst, src, n, dstlen);
  if ((!only_invoke || active) && n >= min_n) { double t1 = now_boottime_s(); log_memcpy_like("MEMMOVE_CHK", dst, src, n, t0, t1, ldprobe_get_invoke_id()); }
  reent--; return ret;
}
