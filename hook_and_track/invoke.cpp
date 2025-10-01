// Invoke bracket wrappers to annotate TFLite C API calls
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <atomic>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <cstdlib>

static inline double now_boottime_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_BOOTTIME, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int log_fd = -1;
static __thread int reent = 0;

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
  if (fd >= 0) log_fd = fd;
  reent--;
}

static void log_line(const char *s) {
  if (log_fd < 0) return;
  size_t n = 0; while (s[n]) ++n;
  (void)write(log_fd, s, n);
  (void)write(log_fd, "\n", 1);
}

static std::atomic<int> g_invoke_id{0};
static __thread int g_invoke_depth = 0;

extern "C" int ldprobe_get_invoke_id(void) {
  return g_invoke_id.load();
}

extern "C" int ldprobe_is_invoke_active(void) {
  return g_invoke_depth > 0;
}

extern "C" void ldprobe_begin_invoke(void) {
  init_log_fd_once();
  const bool outer = (g_invoke_depth == 0);
  if (outer) {
    int id = g_invoke_id.fetch_add(1) + 1;
    double t0 = now_boottime_s();
    char buf[256];
    snprintf(buf, sizeof(buf), "{\"ts\":%.9f,\"ev\":\"INV_BEGIN\",\"invoke\":%d}", t0, id);
    log_line(buf);
  }
  g_invoke_depth++;
}

extern "C" void ldprobe_end_invoke(int status) {
  if (g_invoke_depth <= 0) return;
  const bool outer = (g_invoke_depth == 1);
  g_invoke_depth--;
  if (outer) {
    double t1 = now_boottime_s();
    int id = g_invoke_id.load();
    char buf[256];
    snprintf(buf, sizeof(buf), "{\"ts\":%.9f,\"ev\":\"INV_END\",\"invoke\":%d,\"status\":%d}", t1, id, status);
    log_line(buf);
  }
}

extern "C" int TfLiteInterpreterInvoke(void *interpreter) {
  init_log_fd_once();
  using Fn = int (*)(void*);
  static Fn real = nullptr;
  if (!real) real = (Fn)dlsym(RTLD_NEXT, "TfLiteInterpreterInvoke");
  if (!real) return -1;

  const bool outer = (g_invoke_depth == 0);
  double t0 = 0.0;
  int id_after = 0;
  if (outer) {
    int id = g_invoke_id.fetch_add(1) + 1;
    id_after = id;
    t0 = now_boottime_s();
    char buf[256];
    snprintf(buf, sizeof(buf), "{\"ts\":%.9f,\"ev\":\"INV_BEGIN\",\"invoke\":%d}", t0, id);
    log_line(buf);
  }
  g_invoke_depth++;
  int r = real(interpreter);
  g_invoke_depth--;
  if (outer) {
    double t1 = now_boottime_s();
    char buf[256];
    snprintf(buf, sizeof(buf), "{\"ts\":%.9f,\"ev\":\"INV_END\",\"invoke\":%d,\"status\":%d}", t1, id_after, r);
    log_line(buf);
  }
  return r;
}
