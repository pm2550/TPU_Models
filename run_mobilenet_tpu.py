#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import gc

def load_delegate(lib_path: str):
	try:
		from tflite_runtime import interpreter as tflite
		return tflite, tflite.load_delegate(lib_path)
	except Exception:
		from tensorflow.lite import Interpreter  # type: ignore
		from tensorflow.lite.experimental import load_delegate  # type: ignore
		class _Compat:
			Interpreter = Interpreter
			@staticmethod
			def load_delegate(path):
				return load_delegate(path)
		return _Compat, load_delegate(lib_path)

def create_interpreter_with_delegate(model_path: str, lib_path: str):
	TFL, delegate = load_delegate(lib_path)
	interpreter = TFL.Interpreter(model_path=model_path, experimental_delegates=[delegate])
	interpreter.allocate_tensors()
	return interpreter

def run(model_path: str, lib_path: str, runs: int) -> None:
	TFL, delegate = load_delegate(lib_path)
	interpreter = TFL.Interpreter(model_path=model_path, experimental_delegates=[delegate])
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()[0]
	input_shape = input_details["shape"]
	if input_details["dtype"] == np.uint8:
		input_data = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
	else:
		input_data = np.random.rand(*input_shape).astype(input_details["dtype"])  # type: ignore

	# warmup
	for _ in range(3):
		interpreter.set_tensor(input_details["index"], input_data)
		interpreter.invoke()

	start = time.time()
	for _ in range(runs):
		interpreter.set_tensor(input_details["index"], input_data)
		interpreter.invoke()
	elapsed = (time.time() - start) * 1000.0 / runs

	# 校验进程实际映射的 so 路径，确保选择的库被使用
	actual = ""
	try:
		with open("/proc/self/maps", "r", encoding="utf-8", errors="ignore") as f:
			for line in f:
				if "libedgetpu.so" in line:
					actual = line.strip().split()[-1]
					break
	except Exception:
		pass

	print(f"平均耗时: {elapsed:.2f} ms (runs={runs})  指定库: {lib_path}")
	if actual:
		print(f"实际映射: {actual}")
		if os.path.realpath(actual) != os.path.realpath(lib_path):
			print("警告: 实际加载的库与指定路径不一致")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", required=True, help="tflite 模型路径")
	parser.add_argument("--variant", choices=["std", "max"], default="std")
	parser.add_argument("--lib", default=None, help="自定义 libedgetpu 路径，覆盖 variant")
	parser.add_argument("--runs", type=int, default=20)
	parser.add_argument("--sequence", choices=["std-max", "max-std"], default=None, help="在同一进程内按顺序运行两次推理")
	parser.add_argument("--coexist-test", action="store_true", help="同时持有两个 interpreter（std 与 max）验证能否共存")
	parser.add_argument("--switch-benchmark", choices=["std-max", "max-std"], default=None, help="测量从一种变体切换到另一种的耗时")
	args = parser.parse_args()

	if args.coexist_test:
		lib_std = "/opt/edgetpu/std/libedgetpu.so.1"
		lib_max = "/opt/edgetpu/max/libedgetpu.so.1"
		print(f"模型: {args.model}")
		print("尝试同时创建 std 与 max interpreter...")
		interp_std = None
		interp_max = None
		try:
			interp_std = create_interpreter_with_delegate(args.model, lib_std)
			print("std interpreter 创建成功")
		except Exception as exc:
			print(f"std interpreter 创建失败: {exc}")
		try:
			interp_max = create_interpreter_with_delegate(args.model, lib_max)
			print("max interpreter 创建成功")
		except Exception as exc:
			print(f"max interpreter 创建失败: {exc}")
		# 若均成功，各跑一次
		try:
			if interp_std is not None:
				input_details = interp_std.get_input_details()[0]
				shape = input_details["shape"]
				data = np.random.randint(0, 255, size=shape, dtype=np.uint8)
				interp_std.set_tensor(input_details["index"], data)
				interp_std.invoke()
				print("std invoke 成功")
			if interp_max is not None:
				input_details = interp_max.get_input_details()[0]
				shape = input_details["shape"]
				data = np.random.randint(0, 255, size=shape, dtype=np.uint8)
				interp_max.set_tensor(input_details["index"], data)
				interp_max.invoke()
				print("max invoke 成功")
		except Exception as exc:
			print(f"invoke 失败: {exc}")
		return

	if args.switch_benchmark:
		first, second = args.switch_benchmark.split("-")
		lib1 = f"/opt/edgetpu/{first}/libedgetpu.so.1"
		lib2 = f"/opt/edgetpu/{second}/libedgetpu.so.1"
		print(f"模型: {args.model}")
		# 先创建并跑一次第一种
		print(f"准备 {first} -> {second} 切换基准...")
		interp1 = create_interpreter_with_delegate(args.model, lib1)
		inp1 = interp1.get_input_details()[0]
		shape1 = inp1["shape"]
		data1 = np.random.randint(0, 255, size=shape1, dtype=np.uint8)
		interp1.set_tensor(inp1["index"], data1)
		interp1.invoke()
		print(f"{first} 预热完成")
		# 细粒度计时开始
		all_start = time.time()
		# 步骤1：释放第一种（删除对象+GC）
		gc_start = time.time()
		del interp1
		gc.collect()
		gc_ms = (time.time() - gc_start) * 1000.0
		# 步骤2：加载第二种 delegate
		ld_start = time.time()
		TFL2, delegate2 = load_delegate(lib2)
		ld_ms = (time.time() - ld_start) * 1000.0
		# 步骤3：创建 Interpreter
		crt_start = time.time()
		interp2 = TFL2.Interpreter(model_path=args.model, experimental_delegates=[delegate2])
		crt_ms = (time.time() - crt_start) * 1000.0
		# 步骤4：allocate_tensors
		alloc_start = time.time()
		interp2.allocate_tensors()
		alloc_ms = (time.time() - alloc_start) * 1000.0
		# 步骤5：准备输入数据
		prep_start = time.time()
		inp2 = interp2.get_input_details()[0]
		shape2 = inp2["shape"]
		data2 = np.random.randint(0, 255, size=shape2, dtype=np.uint8)
		prep_ms = (time.time() - prep_start) * 1000.0
		# 步骤6：首次 invoke
		inv_start = time.time()
		interp2.set_tensor(inp2["index"], data2)
		interp2.invoke()
		inv_ms = (time.time() - inv_start) * 1000.0
		all_ms = (time.time() - all_start) * 1000.0
		print(f"切换分步耗时（{first}->{second}）ms:")
		print(f"  GC 回收: {gc_ms:.2f}")
		print(f"  load_delegate: {ld_ms:.2f}")
		print(f"  Interpreter(): {crt_ms:.2f}")
		print(f"  allocate_tensors: {alloc_ms:.2f}")
		print(f"  准备输入: {prep_ms:.2f}")
		print(f"  首次 invoke: {inv_ms:.2f}")
		print(f"  总计: {all_ms:.2f}")
		return

	if args.sequence:
		first, second = args.sequence.split("-")
		lib1 = f"/opt/edgetpu/{first}/libedgetpu.so.1"
		lib2 = f"/opt/edgetpu/{second}/libedgetpu.so.1"
		if not os.path.exists(lib1):
			raise FileNotFoundError(lib1)
		if not os.path.exists(lib2):
			raise FileNotFoundError(lib2)
		print(f"模型: {args.model}")
		print(f"[1/2] 使用 {first}: {lib1}")
		run(args.model, lib1, args.runs)
		# 释放第一轮资源
		gc.collect()
		time.sleep(0.5)
		print(f"[2/2] 使用 {second}: {lib2}")
		run(args.model, lib2, args.runs)
		return

	lib_path = args.lib or f"/opt/edgetpu/{args.variant}/libedgetpu.so.1"
	if not os.path.exists(lib_path):
		raise FileNotFoundError(lib_path)
	print(f"模型: {args.model}")
	print(f"库: {lib_path}")
	run(args.model, lib_path, args.runs)

if __name__ == "__main__":
	main()


