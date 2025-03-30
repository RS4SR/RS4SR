import pickle
import zmq
import logging
import numpy as np
from tqdm import tqdm
import subprocess

from sr import Traffic

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


class Method:
    def __init__(self, name, nc_method, num_cd=None, num_nc=None):
        self.name = name
        self.nc_method = nc_method
        self.num_cd = num_cd
        self.num_nc = num_nc

    def __repr__(self) -> str:
        return f"[name: {self.name} cd: {self.num_cd} nc: {self.num_nc}]"
    
    def filename_suffix(self, cd=False):
        if not cd:
            return f'{self.nc_method}-nc-{self.num_nc}.pkl'
        else:
            return f'{self.nc_method}-nc-{self.num_nc}-cd-{self.num_cd}.pkl'


class LPClient:
    """
    Communicate with server written in rust.
    Send Tasks, and get Solution.
    Data packed with pickle, read/write with zmq ipc

        Task structure:

        # [derive(Debug, Serialize, Deserialize)]
        pub struct Task {
                pub idx: Option<usize>,
                # [serde(flatten)]
                pub cands: Option<Candidates>,
        }

        # [derive(Debug, Serialize, Deserialize)]
        pub struct Candidates {
            pub num_cd: usize,
            pub cands: Cand,
        }

        Solution structure:

        # [derive(Debug, Serialize, Deserialize)]
        pub struct TEResult {
            pub idx: usize,
            pub mlu: f64,
            pub time: f64,
            pub action: Option<Vec<Vec<usize>>>,
        }
    """

    def __init__(self, toponame: str, num_agents: int, id=1,
                 result_dir="./result/", nc_dir='./nc/', 
                 seed=1024, tl=None, num_linkfail=0, num_linkadd=0, num_nodermv=0,
                 is_global_cand=True, cand_path=None, num_thread=1):
        self.id = id
        self.toponame = toponame
        self.result_dir = result_dir
        self.nc_dir = nc_dir
        self.num_agents = num_agents
        self.seed = seed
        self.num_linkfail = num_linkfail
        self.num_linkadd = num_linkadd
        self.num_nodermv = num_nodermv
        self.is_global_cand = is_global_cand
        self.cand_path = cand_path

        self.TMs = Traffic(self.toponame).load_pickle()
        self.push_sock = None
        self.pull_sock = None
        self.tl = tl
        self.num_thread = num_thread

        logger.info(
            f'Run #{self.id} info: topo {self.toponame} with {self.num_agents} agents')

    def set_method(self, method: Method):
        self.method = method
        logger.info(
            f'Set method: {self.method}')

    def split_dataset(self, ratio=0.5):
        """return idxes of splitted data set"""
        num_tm = len(self.TMs)
        idxes = np.arange(num_tm)
        np_state = np.random.RandomState(self.seed)
        np_state.shuffle(idxes)

        len_idxes = len(idxes)
        trainsize = int(ratio*len_idxes)
        trainset, testset = idxes[:trainsize], idxes[trainsize:]
        return trainset, testset
    
    def _gen_rs(self):
        filename = f'{self.nc_dir}/{self.toponame}-{self.method.filename_suffix(cd=True)}'
        f = open(filename, 'rb')
        cands = pickle.load(f)
        return { 'Flow': cands }

    def send_task(self, tm_idx):
        match self.method.nc_method:
            case 'full' | 'opt' | None:
                assert self.method.num_cd is None
                cands = None
            case 'rs':
                cands = self._gen_rs()
            case 'rsp':
                cands = self._gen_rs()
            case 'str':
                cands = self._gen_rs()
            case _:
                assert False

        task = {
            'idx': int(tm_idx),
            'cands': cands,  # cand_levels: 'Network' | 'Node' | 'Flow'
            'num_cd': self.method.num_cd,
        }
        msg = pickle.dumps(task)
        self.push_sock.send(msg)
        return task

    def parse_result(self, result):
        ans = pickle.loads(result)
        idx, mlu, time, sol = ans['idx'], ans['mlu'], ans['time'], ans['action']
        return idx, mlu, time, sol

    def send_quit_signal(self):
        data = {'idx': None, 'cands': None, 'num_cd': None}
        self.push_sock.send(pickle.dumps(data))

    def collect_results(self, tm_idx_set, file_suffix=None, direct_return=False):
        c = zmq.Context()
        self.push_sock = c.socket(zmq.PUSH)
        self.push_sock.bind(f"ipc:///tmp/tasks{self.id}")
        self.pull_sock = c.socket(zmq.PULL)
        self.pull_sock.bind(f"ipc:///tmp/results{self.id}")

        curr = 0  # counter of task

        # push initial tasks
        while curr < min(self.num_agents, len(tm_idx_set)):
            idx = tm_idx_set[curr]
            self.send_task(idx)
            curr += 1

        # pull results
        ids, mlus, times, solutions = [], [], [], []
        bar = tqdm(total=len(tm_idx_set))
        while True:
            result = self.pull_sock.recv()
            if result == b"quit":
                break

            # handle result
            id, mlu, time, sol = self.parse_result(result)
            # print(f'Recv result: [id: {id} mlu: {mlu} time: {time}]')
            ids.append(id), mlus.append(mlu), times.append(
                time), solutions.append(sol)
            bar.update(1)

            # push more tasks
            if curr < len(tm_idx_set):
                idx = tm_idx_set[curr]
                self.send_task(idx)
                curr += 1
            elif curr == len(tm_idx_set):
                self.send_quit_signal()
                curr += 1
            else:
                pass
        bar.close()

        # print analysis info
        maxr, minr, avgr = np.max(mlus), np.min(mlus), np.mean(mlus)
        avg_time = np.mean(times)
        print(f'topology: {self.toponame} method: {self.method}')
        print(f'max:{maxr} min:{minr} avg:{avgr}', flush=True)
        print(f'average solving time: {avg_time:.3f} s', flush=True)

        results = {
            'id': ids,
            'theta': mlus,
            'time': times,
            'solution': solutions
        }
        if direct_return:
            return results
        # save result (if needed)
        if file_suffix is not None:
            f = open(f'{self.result_dir}/{self.toponame}-{file_suffix}', 'wb')
            pickle.dump(results, f)

    def start_server(self, logfile: str = None, need_sol: bool = False):
        """start lpserver"""
        binpath = '/home/wlh/coding/RS4SR2024/lpserver/target/release/lpserver'
        # prepare arguments
        num_nc = '' if self.method.num_nc is None else f'--num-nc={self.method.num_nc}'
        nc_method = '' if self.method.nc_method is None else f'--nc-method={self.method.nc_method}'
        stdout = subprocess.DEVNULL if logfile is None else open(logfile, 'w')
        need_sol = '--need-solution' if need_sol else ''
        nc_dir = f'--nc-dir={self.nc_dir}'
        linkfail = f'--num-linkfail={self.num_linkfail}' if self.num_linkfail > 0 else ''
        linkadd = f'--num-linkadd={self.num_linkadd}' if self.num_linkadd > 0 else ''
        nodermv = f'--num-nodermv={self.num_nodermv}' if self.num_nodermv > 0 else ''
        tl = '' if self.tl is None else f'--time-limit={self.tl}'
        num_thread = '' if self.num_thread is None else f'--num-thread={self.num_thread}'
        is_global_cand = '--is-global-cand' if self.is_global_cand else ''
        cand_path = '' if self.cand_path is None else f'--cand-path={self.cand_path}'
        # start subprocess
        cmd = [f'{binpath}', f'--id={self.id}',
               f'--toponame={self.toponame}', f'--num-agents={self.num_agents}',
               nc_method, num_nc, need_sol, tl, num_thread, nc_dir, linkfail, linkadd, nodermv,
               is_global_cand, cand_path]
        cmd = ' '.join(cmd).strip().split()
        print('server:', ' '.join(cmd))
        subprocess.Popen(cmd, env=None, shell=False,
                         stdout=stdout)
