# spawn LP tasks, collect result and save to file
import logging
import time
from lpclient import LPClient, Method

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


def run(client: LPClient, method: Method, train: bool, slog: bool, need_sol: bool, ratio: float, file_suffix: str):
    client.set_method(method)
    # start server
    logfile = "lpserver.log" if slog else None
    client.start_server(logfile=logfile, need_sol=need_sol)
    time.sleep(1)
    # start client
    trainset, testset = client.split_dataset(ratio=ratio)
    tm_idx_set = trainset if train else testset
    client.collect_results(tm_idx_set, file_suffix)


def collect_optimal(toponame, result_dir='./result/', seed=1024, num_agents=20, num_thread=1):
    # TOPONAMES = ["GEANT", "germany50", "rf1755"]
    client = LPClient(toponame, num_agents=num_agents, id=1,
                        result_dir=result_dir, seed=seed, num_thread=num_thread, is_global_cand=True)
    method = Method(name="optimal", nc_method=None, num_nc=None)
    suffix = f'solution-optimal.pkl'
    run(client, method, train=True, slog=True,
        need_sol=True, ratio=1.0, file_suffix=suffix)


if __name__ == '__main__':
    # collect_optimal(toponame="nobel", num_agents=15)
    # collect_optimal(toponame="rf1221", num_agents=16, num_thread=8)
    # collect_optimal(toponame="rf3967", num_agents=128, num_thread=1)
    # collect_optimal(toponame="rf3257", num_agents=32, num_thread=4)
    # collect_optimal(toponame="rf1755", num_agents=20)
    collect_optimal(toponame="Abilene", num_agents=60)
    # collect_optimal(toponame="germany50", num_agents=80)
    # collect_optimal(toponame="rf1239", num_agents=1)
