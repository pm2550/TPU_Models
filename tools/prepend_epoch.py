#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time


def main() -> int:
    out = sys.stdout
    for line in sys.stdin:
        now = time.time()
        out.write(f"{now:.6f} {line}")
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




