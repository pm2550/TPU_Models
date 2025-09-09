#!/bin/bash
set -euo pipefail
mount -t debugfs none /sys/kernel/debug 2>/dev/null || true
modprobe usbmon || true
mkdir -p /home/10210/Desktop/OS/results/idle_usbmon_5s
SRC=/sys/kernel/debug/usb/usbmon/2u
[ -r  ] || SRC=/sys/kernel/debug/usb/usbmon/0u
# 抓取 5 秒空闲数据
timeout 5s cat  > /home/10210/Desktop/OS/results/idle_usbmon_5s/usbmon_idle.txt || true
# 放权以便后续读取
chown 10210:pm /home/10210/Desktop/OS/results/idle_usbmon_5s/usbmon_idle.txt || true
