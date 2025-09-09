#!/usr/bin/env python3
import argparse
import ctypes
import os
import sys

def load_with_ctypes(library_path: str) -> None:
	print(f"[ctypes] 尝试加载: {library_path}")
	try:
		ctypes.cdll.LoadLibrary(library_path)
		print("[ctypes] 加载成功")
	except OSError as exc:
		print(f"[ctypes] 加载失败: {exc}")
		sys.exit(1)

def try_load_tflite_delegate(library_path: str) -> None:
	try:
		from tflite_runtime import interpreter as tflite
		backend = "tflite_runtime"
	except Exception:
		try:
			from tensorflow.lite import Interpreter as _Interpreter  # type: ignore
			from tensorflow.lite.experimental import load_delegate as _load_delegate  # type: ignore
			class _TFLiteCompat:
				Interpreter = _Interpreter
				@staticmethod
				def load_delegate(lib, options=None):
					return _load_delegate(lib, options or {})
			tflite = _TFLiteCompat()  # type: ignore
			backend = "tensorflow.lite"
		except Exception as exc:
			print(f"[tflite] 未检测到 tflite_runtime 或 tensorflow.lite，跳过 delegate 测试: {exc}")
			return

	print(f"[tflite] 使用后端: {backend}")
	try:
		_delegate = tflite.load_delegate(library_path)
		print("[tflite] load_delegate 成功")
	except Exception as exc:
		print(f"[tflite] load_delegate 失败: {exc}")
		return

def resolve_library_path(variant: str | None, explicit_path: str | None) -> str:
	if explicit_path:
		return explicit_path
	if not variant:
		variant = "std"
	base_dir = f"/opt/edgetpu/{variant}"
	return os.path.join(base_dir, "libedgetpu.so.1")

def main() -> None:
	parser = argparse.ArgumentParser(description="选择性加载 libedgetpu std/max 变体进行测试")
	parser.add_argument("--variant", choices=["std", "max"], default="std", help="选择要测试的变体")
	parser.add_argument("--lib", dest="lib_path", default=None, help="直接指定 libedgetpu.so.1 的绝对路径")
	args = parser.parse_args()

	library_path = resolve_library_path(args.variant, args.lib_path)
	if not os.path.isabs(library_path):
		print(f"错误: 需要绝对路径, 当前: {library_path}")
		sys.exit(2)
	if not os.path.exists(library_path):
		print(f"错误: 文件不存在: {library_path}")
		sys.exit(2)

	print(f"将使用库: {library_path}")
	load_with_ctypes(library_path)
	try_load_tflite_delegate(library_path)
	print("完成")

if __name__ == "__main__":
	main()



