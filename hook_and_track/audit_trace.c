// LD_AUDIT tracer to list dynamically called symbol names and libraries.
// Logs BOOTTIME timestamps to align with invoke brackets recorded by preload.

#define _GNU_SOURCE
#include <link.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static inline double now_boottime_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_BOOTTIME, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int log_fd = -1;
static __thread int reent = 0;

static void init_log(void) {
  if (log_fd >= 0) return;
  if (reent) return;
  reent++;
  const char *p = getenv("LDP_AUDIT_LOG");
  char path[512];
  if (!p || !*p) {
    snprintf(path, sizeof(path), "/tmp/lgaudit_%d.jsonl", (int)getpid());
    p = path;
  }
  int fd = open(p, O_CREAT | O_WRONLY | O_APPEND, 0644);
  if (fd >= 0) log_fd = fd;
  reent--;
}

static void write_line(const char *s) {
  if (log_fd < 0) return;
  size_t n = strlen(s);
  (void)write(log_fd, s, n);
  (void)write(log_fd, "\n", 1);
}

// Simple include filters: comma-separated substrings (e.g., "libedgetpu,libusb")
static const char *include_pat_env = NULL;

// fallback for strnstr on older libc
static const char* my_strnstr(const char *hay, const char *needle, size_t n) {
  size_t ln = needle ? strlen(needle) : 0;
  if (!hay || !needle) return NULL;
  if (ln == 0) return hay;
  for (size_t i = 0; i + ln <= n && hay[i]; ++i) {
    if (strncmp(hay + i, needle, ln) == 0) return hay + i;
  }
  return NULL;
}

static int name_matches(const char *libname) {
  if (!libname) return 0;
  if (!*libname) return 0;
  const char *env = include_pat_env;
  if (!env || !*env) {
    // default include
    return strstr(libname, "libedgetpu") || strstr(libname, "libusb") || strstr(libname, "libtflite") || strstr(libname, "libtensorflowlite");
  }
  const char *p = env;
  while (*p) {
    // read token
    while (*p == ',') ++p;
    if (!*p) break;
    const char *q = strchr(p, ',');
    size_t n = q ? (size_t)(q - p) : strlen(p);
    if (n > 0) {
      if (my_strnstr(libname, p, n) != NULL) return 1;
    }
    if (!q) break; else p = q + 1;
  }
  return 0;
}

typedef struct { uintptr_t cookie; const char *name; } map_rec_t;
static map_rec_t maps[4096];
static int maps_cnt = 0;

static const char *name_for_cookie(uintptr_t *cookie) {
  if (!cookie) return "";
  uintptr_t c = *cookie;
  for (int i = 0; i < maps_cnt; ++i) if (maps[i].cookie == c) return maps[i].name ? maps[i].name : "";
  return "";
}

unsigned int la_version(unsigned int v) {
  init_log();
  include_pat_env = getenv("LDP_AUDIT_INCLUDE");
  return LAV_CURRENT;
}

unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie) {
  init_log();
  (void)lmid;
  if (cookie) *cookie = (uintptr_t)map;
  if (maps_cnt < (int)(sizeof(maps)/sizeof(maps[0]))) {
    maps[maps_cnt].cookie = cookie ? *cookie : 0;
    maps[maps_cnt].name = map && map->l_name ? map->l_name : "";
    maps_cnt++;
  }
  return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

#if __ELF_NATIVE_CLASS == 64
Elf64_Addr la_symbind64(Elf64_Sym *sym, unsigned int symndx, uintptr_t *refcook, uintptr_t *defcook, unsigned int *flags, const char *symname) {
  (void)symndx; (void)flags;
  init_log();
  double t = now_boottime_s();
  const char *to = name_for_cookie(defcook);
  const char *from = name_for_cookie(refcook);
  if (name_matches(to)) {
    char buf[1024];
    snprintf(buf, sizeof(buf), "{\"ts\":%.9f,\"ev\":\"SYMBIND\",\"sym\":\"%s\",\"to\":\"%s\",\"from\":\"%s\"}", t, symname ? symname : "", to, from);
    write_line(buf);
  }
  Elf64_Addr base = 0;
  if (defcook) {
    struct link_map *m = (struct link_map *)(*defcook);
    base = (Elf64_Addr)m->l_addr;
  }
  return base + sym->st_value;
}
#endif

#ifdef __x86_64__
Elf64_Addr la_x86_64_pltenter(Elf64_Sym *sym, unsigned int symndx, uintptr_t *refcook, uintptr_t *defcook, struct La_x86_64_regs *regs, unsigned int *flags, const char *symname, long int *framesizep) {
  (void)symndx; (void)regs; (void)flags; (void)framesizep;
  init_log();
  double t = now_boottime_s();
  const char *to = name_for_cookie(defcook);
  const char *from = name_for_cookie(refcook);
  if (name_matches(to)) {
    char buf[1024];
    snprintf(buf, sizeof(buf), "{\"ts\":%.9f,\"ev\":\"PLT\",\"sym\":\"%s\",\"to\":\"%s\",\"from\":\"%s\"}", t, symname ? symname : "", to, from);
    write_line(buf);
  }
  Elf64_Addr base = 0;
  if (defcook) {
    struct link_map *m = (struct link_map *)(*defcook);
    base = (Elf64_Addr)m->l_addr;
  }
  return base + sym->st_value;
}
#endif

#ifdef __aarch64__
Elf64_Addr la_aarch64_pltenter(Elf64_Sym *sym, unsigned int symndx, uintptr_t *refcook, uintptr_t *defcook, struct La_aarch64_regs *regs, unsigned int *flags, const char *symname, long int *framesizep) {
  (void)symndx; (void)regs; (void)flags; (void)framesizep;
  init_log();
  double t = now_boottime_s();
  const char *to = name_for_cookie(defcook);
  const char *from = name_for_cookie(refcook);
  if (name_matches(to)) {
    char buf[1024];
    snprintf(buf, sizeof(buf), "{\"ts\":%.9f,\"ev\":\"PLT\",\"sym\":\"%s\",\"to\":\"%s\",\"from\":\"%s\"}", t, symname ? symname : "", to, from);
    write_line(buf);
  }
  Elf64_Addr base = 0;
  if (defcook) {
    struct link_map *m = (struct link_map *)(*defcook);
    base = (Elf64_Addr)m->l_addr;
  }
  return base + sym->st_value;
}
#endif
