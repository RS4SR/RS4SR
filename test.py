import time
from fire import Fire
from lpclient import LPClient, Method


def run(
    client: LPClient,
    method: Method,
    train: bool,
    slog: bool,
    need_sol: bool,
    ratio: float,
    file_suffix: str,
):
    client.set_method(method)
    # start server
    logfile = "lpserver.log" if slog else None
    client.start_server(logfile=logfile, need_sol=need_sol)
    time.sleep(1)
    # start client
    trainset, testset = client.split_dataset(ratio=ratio)
    tm_idx_set = trainset if train else testset
    client.collect_results(tm_idx_set, file_suffix)


def test(
    toponame,
    method,
    num_nc,
    num_cd,
    time,
    num_agents,
    result_dir="./result",
    nc_dir="./nc",
    is_global_cand=True,
    cand_path=None,
):
    """Recommender System for SR"""
    method = Method(name=method, nc_method=method, num_cd=num_cd, num_nc=num_nc)
    client = LPClient(
        toponame,
        num_agents=num_agents,
        id=1,
        tl=time,
        result_dir=result_dir,
        nc_dir=nc_dir,
        is_global_cand=is_global_cand,
        cand_path=cand_path,
    )
    suffix = (
        f"result-{method.nc_method}-nc-{method.num_nc}-cd-{method.num_cd}-t{time}.pkl"
    )
    run(
        client,
        method,
        train=False,
        slog=False,
        need_sol=False,
        ratio=0.7,
        file_suffix=suffix,
    )


def main(
    toponame,
    method,
    num_nc,
    num_cd,
    time,
    result_dir,
    nc_dir,
    num_agents,
    is_global_cand,
    cand_path,
):
    test(
        toponame,
        method,
        num_nc,
        num_cd,
        time,
        num_agents=num_agents,
        result_dir=result_dir,
        nc_dir=nc_dir,
        is_global_cand=is_global_cand,
        cand_path=cand_path,
    )


if __name__ == "__main__":
    Fire(main)

    # test(toponame="nobel", method="rs", num_nc=17, num_cd=5, time=1, num_agents=20)
    # test(toponame="nobel", method="rsp", num_nc=5, num_cd=5, time=1, num_agents=20)
    #
    # test(toponame="GEANT", method="rs", num_nc=22, num_cd=5, time=1, num_agents=20)
    # test(toponame="GEANT", method="rsp", num_nc=5, num_cd=5, time=1, num_agents=20)

    # test(toponame="rf1755", method="rs", num_nc=87, num_cd=5, time=1, num_agents=20)
    # test(toponame="rf1755", method="rsp", num_nc=5, num_cd=5, time=1, num_agents=20)

    # test(toponame="rf6461", method="rs", num_nc=138, num_cd=5, time=1, num_agents=20)
    # test(toponame="rf6461", method="rsp", num_nc=5, num_cd=5, time=2, num_agents=20)
