#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pycoral.utils.edgetpu import list_edge_tpus


def main():
    buses = []
    for d in list_edge_tpus():
        t = (d.get('type') or '').lower()
        if t != 'usb':
            continue
        path = d.get('path') or ''  # e.g., /sys/bus/usb/devices/1-1
        m = re.search(r'/devices/(\d+)-', path)
        if m:
            buses.append(int(m.group(1)))
    buses = sorted(set(buses))
    print(json.dumps({'buses': buses}, ensure_ascii=False))


if __name__ == '__main__':
    main()


