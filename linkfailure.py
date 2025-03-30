from lpclient import LPClient, Method
from fire import Fire
from copy import deepcopy
import time
import pickle
import numpy as np
import zmq
import random
import networkx as nx

from sr import Topology

def gen_failed_links(toponame, num_linkfail: int):
    """randomly remove some links"""
    if num_linkfail <= 0:
        # do nothing
        return

    ok = False
    G = Topology(toponame).load()
    while not ok:
        # select links
        idxes = list(range(G.number_of_edges()))
        idxes = random.sample(idxes, k=num_linkfail)
        idxes.sort()
        p = 0
        result = []
        for i, (src, dst, _) in enumerate(G.edges.data()):
            if p >= len(idxes):
                break
            if i == idxes[p]:
                result.append((src, dst))
                p += 1
        assert num_linkfail == len(result)
        # ensure G is connected
        tmpG = deepcopy(G)
        for (src, dst) in result:
            tmpG.remove_edge(src, dst)
        if not nx.is_strongly_connected(tmpG):
            continue
        # found one
        ok = True
        print(f'link {idxes} failed')
    return result


def send_failed_links(id, failed_links):
    # send to client
    c = zmq.Context()
    push_sock = c.socket(zmq.PUSH)
    push_sock.bind(f"ipc:///tmp/link_failures{id}")

    msg = pickle.dumps(failed_links)
    push_sock.send(msg)


def run_with_failed_links(id: int, client: LPClient, method: Method, train: bool, slog: bool, need_sol: bool, ratio: float, file_suffix: str,
        num_tm_each_run: int, failed_links, seed: int):
    client.set_method(method)
    # start server
    logfile = "lpserver.log" if slog else None
    client.start_server(logfile=logfile, need_sol=need_sol)
    time.sleep(1)
    # start client
    trainset, testset = client.split_dataset(ratio=ratio)

    np_state = np.random.RandomState(seed)
    testset = np_state.choice(testset, size=num_tm_each_run, replace=False)

    # 3. send failed links to server
    send_failed_links(id, failed_links)

    results = client.collect_results(testset, file_suffix, direct_return=True)
    time.sleep(1)
    return results


def test(toponame, method, num_nc, num_cd, time, 
         num_agents=20, id=1, seed=1024, alpha=1.3, save=True,
         num_linkfail=0, num_tm_each_run=20, run_times=1,
         result_dir="./result/", nc_dir="./nc/"):
    # method = Method(cd_method, num_inodes, num_tnodes, lp_kind, obj_kind)
    method = Method(name=method, nc_method=method, num_cd=num_cd, num_nc=num_nc)
    client = LPClient(toponame, num_agents=num_agents, id=id, tl=time, result_dir=result_dir, nc_dir=nc_dir, num_linkfail=num_linkfail)
    file_suffix = f'result-{method.nc_method}-nc-{method.num_nc}-cd-{method.num_cd}-t{time}-linkfail-{num_linkfail}.pkl'

    # do multiple runs each with 
    results_array = []
    all_avg = 0
    random.seed(seed)
    for i in range(run_times):
        # robust scenerio
        failed_links = gen_failed_links(toponame=toponame, num_linkfail=num_linkfail)

        results = run_with_failed_links(id, client, method, train=False, slog=False, need_sol=False, 
                                        ratio=0.7, file_suffix=file_suffix, 
                                        num_tm_each_run=num_tm_each_run, failed_links=failed_links, 
                                        seed=seed)

        results_array.append(results)

        # print analysis info
        objs, times = results['theta'], results['time']
        maxr, minr, avgr = np.max(objs), np.min(objs), np.mean(objs)
        avg_time = np.mean(times)
        print(f'topology: {toponame} method: {method} num_linkfail: {num_linkfail}')
        print(f'max:{maxr} min:{minr} avg:{avgr}', flush=True)
        print(f'average solving time: {avg_time:.3f} s', flush=True)
        all_avg += (avgr / run_times)

    print('===================================================================')
    print(f'topology: {toponame} method: {method} num_linkfail: {num_linkfail}')
    print(f'avg result: {all_avg}')
    print('===================================================================')

    # save result (if needed)
    if save:
        f = open(f'{result_dir}/{toponame}-{file_suffix}', 'wb')
        pickle.dump(results_array, f)


if __name__ == '__main__':
    Fire(test)
