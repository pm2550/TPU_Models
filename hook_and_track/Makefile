CC=gcc
CXX=g++
CFLAGS=-O2 -g -fPIC -Wall -Wextra -D_GNU_SOURCE
CXXFLAGS=-O2 -g -fPIC -Wall -Wextra -D_GNU_SOURCE
LDFLAGS=-shared -ldl -lpthread

OUT_PRELOAD=libldprobe_gate.so
OUT_AUDIT=liblgaudit.so
SRC_C=ldprobe_gate.c
SRC_CXX=invoke.cpp
SRC_AUD=audit_trace.c

all: $(OUT_PRELOAD) $(OUT_AUDIT)

$(OUT_PRELOAD): $(SRC_C) $(SRC_CXX)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_C) $(SRC_CXX) $(LDFLAGS)

$(OUT_AUDIT): $(SRC_AUD)
	$(CC) $(CFLAGS) -shared -o $@ $(SRC_AUD) -ldl

clean:
	rm -f $(OUT_PRELOAD) $(OUT_AUDIT) *.o

.PHONY: all clean
