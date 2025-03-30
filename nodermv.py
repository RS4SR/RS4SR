# spawn LP tasks, collect result and save to file
import logging
import time
from lpclient import LPClient, Method
from itertools import product
from sr import Topology
import random
import numpy as np
import zmq
import pickle
import networkx as nx
from copy import deepcopy

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)

def send_removed_nodes(id, removed_nodes):
    # send to client
    c = zmq.Context()
    push_sock = c.socket(zmq.PUSH)
    push_sock.bind(f"ipc:///tmp/node_remove{id}")

    msg = pickle.dumps(removed_nodes)
    push_sock.send(msg)

def gen_removed_nodes(toponame, seed, num_nodermv: int) -> list | None:
    """randomly add some links"""
    if num_nodermv <= 0:
        # do nothing
        return None

    G = Topology(toponame).load()

    random.seed(seed)
    ok = False
    while not ok:
        idxes = list(range(G.number_of_nodes()))
        random.shuffle(idxes)
        removed_nodes = idxes[:num_nodermv]
        tmpG = deepcopy(G)
        for node in removed_nodes:
            tmpG.remove_node(node)
        if not nx.is_strongly_connected(tmpG):
            continue
        ok = True

    # select links
    assert len(removed_nodes) == num_nodermv
    print(f"node {removed_nodes} removed")
    return removed_nodes

def run(client: LPClient, method: Method, train: bool, slog: bool, need_sol: bool, ratio: float, file_suffix: str, num_nodermv: int, seed):
    client.set_method(method)
    # start server
    logfile = "lpserver.log" if slog else None
    client.start_server(logfile=logfile, need_sol=need_sol)
    time.sleep(1)
    # start client
    trainset, testset = client.split_dataset(ratio=ratio)
    tm_idx_set = trainset if train else testset


    if num_nodermv > 0:
        removed_nodes = gen_removed_nodes(client.toponame, seed, num_nodermv)
        send_removed_nodes(client.id, removed_nodes)

    client.collect_results(tm_idx_set, file_suffix)


def collect_nodermv(toponame, result_dir='./result/', seed=1024, num_agents=20, num_thread=1, num_nodermv=0, rmv_seed=42):
    # TOPONAMES = ["GEANT", "germany50", "rf1755"]
    client = LPClient(toponame, num_agents=num_agents, id=10000 + rmv_seed,
                        result_dir=result_dir, seed=seed, num_thread=num_thread, is_global_cand=True, num_nodermv=num_nodermv)
    method = Method(name="optimal", nc_method=None, num_nc=None)
    suffix = f'solution-optimal-rmv-{num_nodermv}-seed-{rmv_seed}.pkl'
    run(client, method, train=True, slog=True,
        need_sol=True, ratio=0.1, file_suffix=suffix, num_nodermv=num_nodermv, seed=rmv_seed)


if __name__ == '__main__':
    for toponame in ["nobel", "GEANT", "rf1755"]:
        for num_nodermv in [1, 2, 3, 4]:
            for seed in range(42, 42+10):
                print(f"{toponame}-{num_nodermv}-{seed}:")
                collect_nodermv(toponame=toponame, num_agents=28, result_dir="./result.rmv", num_nodermv=num_nodermv, rmv_seed=seed)
                time.sleep(1)
