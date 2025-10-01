import numpy as np, time, os
m = r"…"
it=None
try:
    from pycoral.utils.edgetpu import make_interpreter
    it=make_interpreter(m)
except Exception:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    it=Interpreter(model_path=m, experimental_delegates=[load_delegate('libedgetpu.so.1')])
it.allocate_tensors()
inp=it.get_input_details()[0]
x = np.random.randint(0,256, size=inp['shape'], dtype=np.uint8 if inp['dtype'].__name__=='uint8' else np.int8)
import ctypes
lib = ctypes.CDLL(None)
begin = getattr(lib, 'ldprobe_begin_invoke', None)
end = getattr(lib, 'ldprobe_end_invoke', None)
if begin is not None: begin.restype=None
if end is not None: end.argtypes=[ctypes.c_int]; end.restype=None
tail_ms = float(os.environ.get('INV_TAIL_MS','0') or '0')
for _ in range(2):
    if begin is not None: begin()
    it.set_tensor(inp['index'], x)
    it.invoke()
    _=it.get_tensor(it.get_output_details()[0]['index'])
    time.sleep(tail_ms/1000.0)
    if end is not None: end(0)
for i in range(int(os.environ.get('COUNT','10'))):
    if begin is not None: begin()
    it.set_tensor(inp['index'], x)
    it.invoke()
    _=it.get_tensor(it.get_output_details()[0]['index'])
    time.sleep(tail_ms/1000.0)
    if end is not None: end(0)
