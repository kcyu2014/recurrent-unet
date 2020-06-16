import datetime
import linecache
import os

import pynvml
import torch

print_tensor_sizes = True
last_tensor_sizes = set()
gpu_profile_fn = f'{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-gpu_mem_prof.txt'
if 'GPU_DEBUG' in os.environ:
    print('profiling gpu usage to ', gpu_profile_fn)

lineno = 1
func_name = 'loss'
filename = 'train'
module_name = 'cityscape_pspnet'
os.environ['GPU_DEBUG'] = '2'


def gpu_profile(frame, event, arg):
    print_tensor_sizes = True
    last_tensor_sizes = set()
    gpu_profile_fn = f'{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-gpu_mem_prof.txt'
    if 'GPU_DEBUG' in os.environ:
        print('profiling gpu usage to ', gpu_profile_fn)

    # it is _about to_ execute (!)
    # global last_tensor_sizes
    # global lineno, func_name, filename, module_name

    lineno = 1
    func_name = 'loss'
    filename = 'train'
    module_name = 'cityscape_pspnet'
    os.environ['GPU_DEBUG'] = '2'

    if event == 'line':
        try:
            # about _previous_ line (!)
            if lineno is not None:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ['GPU_DEBUG']))
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                line = linecache.getline(filename, lineno)
                where_str = module_name+' '+func_name+':'+str(lineno)

                with open(gpu_profile_fn, 'a+') as f:
                    f.write(f"{where_str:<50}"
                            f":{meminfo.used/1024**2:<7.1f}Mb "
                            f"{line.rstrip()}\n")

                    if print_tensor_sizes is True:
                        for tensor in get_tensors():
                            if not hasattr(tensor, 'dbg_alloc_where'):
                                tensor.dbg_alloc_where = where_str
                        new_tensor_sizes = {(type(x), tuple(x.size()), x.dbg_alloc_where)
                                            for x in get_tensors()}

                        for t, s, loc in new_tensor_sizes:
                            f.write(f'+ {loc},{str(s)},{str(t)}\n')
                        # for t, s, loc in new_tensor_sizes - last_tensor_sizes:
                        #     f.write(f'+ {loc:<50} {str(s):<20} {str(t):<10}\n')
                        # for t, s, loc in last_tensor_sizes - new_tensor_sizes:
                        #     f.write(f'- {loc:<50} {str(s):<20} {str(t):<10}\n')
                        # last_tensor_sizes = new_tensor_sizes
                pynvml.nvmlShutdown()

            # save details about line _to be_ executed
            # lineno = None

            func_name = frame.f_code.co_name
            filename = frame.f_globals["__file__"]
            if (filename.endswith(".pyc") or
                    filename.endswith(".pyo")):
                filename = filename[:-1]
            module_name = frame.f_globals["__name__"]
            lineno = frame.f_lineno

            if 'gmwda-pytorch' not in os.path.dirname(os.path.abspath(filename)):
                lineno = None  # skip current line evaluation

            if ('car_datasets' in filename
                    or '_exec_config' in func_name
                    or 'gpu_profile' in module_name
                    or 'tee_stdout' in module_name):
                lineno = None  # skip current

            return gpu_profile

        except (KeyError, AttributeError):
            pass

    return gpu_profile


def get_tensors(gpu_only=True):
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        except Exception as e:
            pass