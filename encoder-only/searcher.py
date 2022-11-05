from typing import Union
from gpu_utils import getGPUs, getAvailabilityGPU
import sys
import random
from itertools import product
import json
from datetime import datetime
import subprocess
import time
import os
import signal
import threading
import warnings


class Searcher:
    def __init__(self, gpus, python_fn, root_dir, gpu_for_hyper:Union[int, callable], num_trials=-1, repeat=1):
        """

        :param gpus: -1, 所有可用的gpu; 数字，指定gpu; list[int]包含的所有的gpu。
        :param python_fn: 需要运行的python文件名
        :param root_dir: 传入一个文件夹指示在哪里暂存search中间文件
        :param gpu_for_hyper: 如果为int，指示的是每个gpu可以跑多少组实验; 如果为callable，将传入(1){}，其中包含所有的hyper，
            (2) 本程序在每个uuid的gpu上已经跑的程序的uuid (3) root_dir, 存储每个uuid的程序运行状态的文件夹
            需要返回使用的gpu的值(str类型)，多卡就是'2,3'
        :param num_trials: 实验多少次，仅在输入了random search参数时有效. 而且是random search会采多少次
        :param repeat: 每组参数跑多少次实验
        """
        assert os.path.exists(python_fn)
        self.python_fn = python_fn

        GPUs = getGPUs()
        if gpus == -1:
            self.gpus = GPUs
        elif isinstance(gpus, int):
            self.gpus = [_gpu for _gpu in GPUs if _gpu.id==gpus]
        elif isinstance(gpus, list):
            self.gpus = []
            for gpu_id in gpus:
                for _gpu in GPUs:
                    if _gpu.id == gpu_id:
                        self.gpus.append(_gpu)
                        break
            # self.gpus = [_gpu for _gpu in GPUs if _gpu.id in gpus]
        else:
            raise RuntimeError("Unsupport type.")
        if len(self.gpus) == 0:
            raise RuntimeError(f"{gpus} is not a valid gpu")

        self.gpu_for_hyper = gpu_for_hyper
        self.assigned_gpus = {gpu.uuid:[] for gpu in self.gpus}  # 被分配了的gpu，key是gpu的id，value是list[uuid],value中的uuid是程序的uuid

        os.makedirs(root_dir, exist_ok=True)
        self.root_dir = root_dir
        if not os.path.exists(os.path.join(self.root_dir, 'meta.tsv')):
            with open(os.path.join(self.root_dir, 'meta.tsv'), 'w', encoding='utf-8') as f:  # 创建空文件
                #  TODO 新增一个线程号，用于关闭
                f.write("{}\t{}\t{}\t{}\n".format('UUID', 'device', 'fp', 'status'))
        self.grid_search_params = {}  # 每一次random search sample出来的结果都需要一一配对
        self.random_search_params = {}
        self.sequential_search_params = []  # 依次搜索的参数
        if num_trials == -1:
            num_trials = sys.maxsize
        self.num_trials = num_trials
        self.repeat = repeat

    def is_param_valid(self, **kwargs):
        """
        判断新加入的参数是否是合理的。比如在random search 中出现的参数就不能在grid中出现，反之亦然

        :param kwargs:
        :return:
        """
        for key, value in kwargs.items():
            if isinstance(value, list) or callable(value):
                pass
            else:
                value = [value]
            for v in value:
                assert isinstance(v, (float, int, bool, str))
            assert key not in self.random_search_params
            assert key not in self.grid_search_params

    def add_grid_search_params(self, **kwargs):
        """
        加入的内容必须是int, bool, str, float类型的。

        :param kwargs:
        :return:
        """
        self.is_param_valid(**kwargs)
        self.grid_search_params.update(kwargs)

    def add_random_search_params(self, **kwargs):
        """
        每次从kwargs中的结合中随机选择。如果有grid_search的结果，则任何一组表示都和grid_search会一一check，必须要float,
            str, bool, int。也可以是callallbe的，不能带任何参数

        :param kwargs:
        :return:
        """
        self.is_param_valid(**kwargs)
        self.random_search_params.update(kwargs)

    def add_sequential_params(self, **kwargs):
        """
        优先搜寻sequential的参数, 如果某个参数只有一个值，只有一个值的则broadcast到所有值。如果为list的值则所有值都一一broadcast。

        :param kwargs:
        :return:
        """
        L = -1
        for key, value in kwargs.items():
            if isinstance(value, list):
                L = len(value) if L==-1 else L
                if L!=len(value):
                    raise RuntimeError(f"{key} has different number of params.")

        L = 1 if L==-1 else L

        params = [{} for _ in range(L)]
        for key, value in kwargs.items():
            if isinstance(value, list):
                for param, v in zip(params, value):
                    assert isinstance(v, (float, int, bool, str))
                    param[key] = v
            else:
                for param in params:
                    assert isinstance(value, (float, int, bool, str))
                    param[key] = value

        self.sequential_search_params = params

    def notify_finish(self, uuid):
        """
        指示uuid这个实验已经运行完毕

        :param uuid:
        :return:
        """
        empty = True
        with open(os.path.join(self.root_dir, uuid, 'err.txt'), 'r') as f:
            for line in f:
                line.strip()
                if line:
                    empty = False
                    break

        with open(os.path.join(self.root_dir, 'meta.tsv'), 'r+', encoding='utf-8') as f:
            line = f.readline()
            while line:
                if line.startswith(uuid):
                    index = f.tell()
                    f.seek(index-len('running   \n'))
                    if not empty:
                        f.write("error".ljust(len('running   '), ' '))
                    else:
                        f.write('finish'.ljust(len('running   '), ' '))
                    break
                line = f.readline()

        for gpu_uuid, pros in self.assigned_gpus.items():
            if uuid in pros:
                pros.remove(uuid)

    def get_next_hyper(self):
        """
        产生下一组超参搜索。key是参数，value是对应的值

        :return: {}
        """

        # TODO 修改为每次从文件中check新的parameter
        # TODO 允许使用function判断是否会输出某个hyper
        # fcntl

        if len(self.sequential_search_params)==0 and len(self.random_search_params)==0 and len(self.grid_search_params)==0:
            raise RuntimeError("You have to specify param first.")

        for hyper in self.sequential_search_params:
            for _ in range(self.repeat):
                yield hyper

        if len(self.grid_search_params):
            grid_params = []
            for param in self.grid_search_params.values():
                if isinstance(param, list):
                    grid_params.append(param)
                else:
                    grid_params.append([param])


        if len(self.random_search_params)>0:
            while self.num_trials > 0:
                self.num_trials -= 1
                hyper = {}
                for key, value in self.random_search_params.items():
                    if isinstance(value, list):
                        value = random.choice(value)
                    elif callable(value):
                        value = value()
                    hyper[key] = value
                if len(self.grid_search_params):
                    for grids in product(*grid_params):
                        for key, grid in zip(self.grid_search_params.keys(), grids):
                            hyper[key] = grid
                        for _ in range(self.repeat):
                            yield hyper
                else:
                    for _ in range(self.repeat):
                        yield hyper
        else:
            if len(self.grid_search_params):
                hyper = {}
                grid_params = list(product(*grid_params))
                for grids in grid_params:
                    for key, grid in zip(self.grid_search_params.keys(), grids):
                        hyper[key] = grid
                    for _ in range(self.repeat):
                        yield hyper

    def get_next_device(self, hyper):
        """
        返回hyper这个参数在哪个device上运行

        :param hyper:
        :return: 如果返回值为None，说明目前还没有合适的显卡
        """
        if isinstance(self.gpu_for_hyper, int):
            for gpu in self.gpus:
                if len(self.assigned_gpus[gpu.uuid])<self.gpu_for_hyper and gpu.memoryUtil<1/self.gpu_for_hyper:
                    return str(gpu.id)
        else:
            return str(self.gpu_for_hyper(hyper, self.assigned_gpus.copy(), self.root_dir))
        return None

    def __iter__(self):
        """
        一次迭代一个subprocess的命令行。

        :return:
        """
        for hyper in self.get_next_hyper():
            if hyper is None:
                break
            print(f"Get hyper:{hyper}")
            cuda_vis_de = self.get_next_device(hyper)
            while cuda_vis_de is None:  # 等待可以用的显卡
                cuda_vis_de = self.get_next_device(hyper)
                time.sleep(1)
            uuid = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            while os.path.exists(os.path.join(self.root_dir, uuid)):
                time.sleep(1)
                uuid = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            gpu_ids = list(map(int, cuda_vis_de.split(',')))
            for gpu_id in gpu_ids:
                gpu = [gpu for gpu in self.gpus if gpu.id == gpu_id][0]
                self.assigned_gpus[gpu.uuid].append(uuid)
            uuid_dir = os.path.join(self.root_dir, uuid)
            os.makedirs(uuid_dir, exist_ok=True)
            with open(os.path.join(uuid_dir, 'meta.txt'), 'w', encoding='utf-8') as f:
                f.write(json.dumps(hyper) + '\n')
                f.write(cuda_vis_de)
            output_fp = os.path.join(uuid_dir, 'output.txt')
            err_fp = os.path.join(uuid_dir, 'err.txt')
            hyper_line = ''
            for key, value in hyper.items():
                hyper_line += '--{} {} '.format(key, value)
            # -u 是为了flush https://stackoverflow.com/questions/107705/disable-output-buffering
            cmd = 'CUDA_VISIBLE_DEVICES={} python -u {} {}'.format(cuda_vis_de, self.python_fn, hyper_line)
            #  记录一笔
            with open(os.path.join(self.root_dir, 'meta.tsv'), 'a', encoding='utf-8') as f:  # 创建空文件
                f.write('{}\t{}\t{}\t{}\n'.format(uuid, cuda_vis_de, self.python_fn, 'running   '))

            yield cmd, output_fp, err_fp, uuid

        print("Finish generating hyper.")

        # raise StopIteration


class Runner:
    """
    一次运行一个命令行的命令，并且将结果指示给searcher。所有的命令仅支持python args传入。

    """
    def __init__(self, searcher):
        self.searcher = searcher
        self.keep_alives = {}
        self.stop_processes = {}
        self.pids = {}

    def start_new_run(self, cmd, std_file, err_file, uuid):
        with open(std_file, 'w') as f1, open(err_file, 'w') as f2:
            print(f"Start run {cmd}\n output in {std_file}\n")
            popen = subprocess.Popen(cmd, shell=True, stdout=f1, stderr=f2,
                                     preexec_fn=os.setsid)
            while popen.poll() is None:
                if self.keep_alives[uuid]:
                    time.sleep(1)
                else:
                    os.killpg(popen.pid, signal.SIGINT)
            self.stop_processes[uuid] = True
            self.pids[uuid] = popen.pid
        self.notify_finish(uuid)

    def clear_thread(self):
        """
        清除已经运行结束的程序

        :return:
        """
        for uuid in list(self.stop_processes.keys()):
            stop_process = self.stop_processes[uuid]
            if stop_process:
                flag = self.stop_uuid(uuid)
                if flag:
                    print(f"Clear {uuid}.")
                else:
                    print(f"Fail to clear {uuid}")

    def notify_finish(self, uuid):
        self.searcher.notify_finish(uuid)

    def run(self):
        self.threads = {}
        try:
            for cmd, std_file, err_file, uuid in self.searcher:
                self.clear_thread()
                self.keep_alives[uuid] = True
                self.stop_processes[uuid] = False
                t = threading.Thread(target=self.start_new_run, args=(cmd, std_file, err_file, uuid))
                t.start()
                self.threads[uuid] = t
            for t in self.threads.values():
                t.join()

        except BaseException as e:
            raise e
        finally:
            self.stop()

    def stop_uuid(self, uuid):
        """
        返回关闭成功还是失败
        :param uuid:
        :return: bool
        """
        self.keep_alives[uuid] = False
        wait_count = 0
        while True or wait_count < 10:
            wait_count += 1
            if self.stop_processes[uuid]:
                self.stop_processes.pop(uuid)
                self.pids.pop(uuid)
                self.threads.pop(uuid)
                self.keep_alives.pop(uuid)
                return True
            time.sleep(1)
        return False

    def stop(self):
        for uuid in list(self.threads.keys()):
            self.stop_uuid(uuid)
        if len(self.pids):
            print("The following pids are not terminated.{}".format(self.pids.values()))
        else:
            print("All pid have been cleared.")


class HyperSearch:
    def __init__(self, gpus, python_fn, root_dir='search/', gpu_for_hyper:Union[int, callable]=1, num_trials=100,
                 repeat=1):
        """

        :param gpus: -1, 所有可用的gpu; 数字，指定gpu; list[int]包含的所有的gpu。
        :param python_fn: 需要运行的python文件名
        :param root_dir: 传入一个文件夹指示在哪里暂存search中间文件
        :param gpu_for_hyper: 如果为int，指示的是每个gpu可以跑多少组实验; 如果为callable，将传入(1){}，其中包含所有的hyper，
            (2) 本程序在每个uuid的gpu上已经跑的程序的uuid (3) root_dir, 存储每个uuid的程序运行状态的文件夹
            需要返回使用的gpu的值(str类型)，多卡就是'2,3'
        :param num_trials: 实验多少次，仅在输入了random search参数时有效
        :param repeat: 每个实验重复多少次
        """
        #
        with open(python_fn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('fitlog.debug()'):
                    warnings.warn("You are in debug mode.")
                if line.startswith('fitlog.commit'):
                    raise RuntimeError("You should not commit in your python script.")
        self.searcher = Searcher(gpus, python_fn, root_dir, gpu_for_hyper, num_trials, repeat)
        self.runner = Runner(self.searcher)

    def start_search(self):
        self.runner.run()

    def stop(self):
        self.runner.stop()