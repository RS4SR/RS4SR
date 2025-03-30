import pickle
import numpy as np

NOP = -1
SP = -2

NAME_MAP = {
    "tnsm": "DeepLS",
    "icnp": "MARL-GNN",
    "defo": "DEFO",
    "srls": "SRLS",
    "ours": "Ours",
    "opt": "ILP",
    "str": "STR",
}

def split_dataset(length: int, ratio: float, seed):
    """return idxes of splitted data set"""
    idxes = np.arange(length)
    np_state = np.random.RandomState(seed)
    np_state.shuffle(idxes)

    len_idxes = len(idxes)
    trainsize = int(ratio*len_idxes)
    trainset, testset = idxes[:trainsize], idxes[trainsize:]
    return trainset, testset


def transform_action_matrix(mat: np.array, n: int) -> np.array:
    """given a NxN matrix, replace shortest path to a special value"""
    for i in range(n):
        for j in range(n):
            if mat[i][j] == i or mat[i][j] == j:
                mat[i][j] = SP
    return mat


def load_dataset(toponame: str, result_dir: str = "./result") -> tuple[np.array, int]:
    result_file = open(f"{result_dir}/{toponame}-solution-optimal.pkl", "rb")
    result = pickle.load(result_file)
    ids = result["id"]
    thetas, times = result["theta"], result["time"]
    sols = result["solution"]
    num_nodes = len(sols[0])

    dataset = []
    for id, theta, time, sol in zip(ids, thetas, times, sols):
        sol = np.array(sol)
        sol = transform_action_matrix(sol, len(sol))
        # print(id, theta, time, sol.shape)
        dataset.append(sol.flatten())
    dataset = np.array(dataset).T  # shape: [num_flows, num_samples]
    return dataset, num_nodes


def load_dataset_split(toponame: str, result_dir: str ="./result", ratio=0.7) -> tuple[np.array, int]:
    result_file = open(f"{result_dir}/{toponame}-solution-optimal.pkl", "rb")
    result = pickle.load(result_file)
    ids = result["id"]
    thetas, times = result["theta"], result["time"]
    sols = result["solution"]
    num_nodes = len(sols[0])

    train_idx, test_idx = split_dataset(len(ids), ratio, seed=1024)
    train_idx, test_idx = set(train_idx), set(test_idx)

    trainset = []
    testset = []
    for id, theta, time, sol in zip(ids, thetas, times, sols):
        if id in train_idx:
            sol = np.array(sol)
            sol = transform_action_matrix(sol, len(sol))
            # print(id, theta, time, sol.shape)
            trainset.append(sol.flatten())
        else:
            assert id in test_idx
            sol = np.array(sol)
            sol = transform_action_matrix(sol, len(sol))
            # print(id, theta, time, sol.shape)
            testset.append(sol.flatten())
    trainset = np.array(trainset).T  # shape: [num_flows, num_samples]
    testset = np.array(testset).T  # shape: [num_flows, num_samples]
    return trainset, testset, num_nodes


import itertools
from matplotlib.colors import hex2color

# function copy from seaborn to generate n unique_dashes
def unique_dashes(n):
    # Start with dash specs that are well distinguishable
    dashes = [
        "",
        (4, 1.5),
        (1, 1),
        (3, 1.25, 1.5, 1.25),
        (5, 1, 1, 1),
    ]
    dashes = dashes[::-1]

    # Now programatically build as many as we need
    p = 3
    while len(dashes) < n:
        # Take combinations of long and short dashes
        a = itertools.combinations_with_replacement([3, 1.25], p)
        b = itertools.combinations_with_replacement([4, 1], p)
        # Interleave the combinations, reversing one of the streams
        segment_list = itertools.chain(*zip(
            list(a)[1:-1][::-1],
            list(b)[1:-1]
        ))
        # Now insert the gaps
        for segments in segment_list:
            gap = min(segments)
            spec = tuple(itertools.chain(*((seg, gap) for seg in segments)))
            dashes.append(spec)
        p += 1
    return dashes[:n]

# color schemes Courtesy of the excellent mbostock's D3.js project
d3_10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

d3_20 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

d3_20b = ['#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39', '#e7ba52', '#e7cb94', '#843c39', '#ad494a', '#d6616b', '#e7969c', '#7b4173', '#a55194', '#ce6dbd', '#de9ed6']

d3_20c = ['#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354', '#74c476', '#a1d99b', '#c7e9c0', '#756bb1', '#9e9ac8', '#bcbddc', '#dadaeb', '#636363', '#969696', '#bdbdbd', '#d9d9d9']

colors10  = list(map(hex2color, d3_10))
colors20  = list(map(hex2color, d3_20))
colors20b = map(hex2color, d3_20b)
colors20c = map(hex2color, d3_20c)


# color schemes Mathematica
colour_01 = (0.368417, 0.506779, 0.709798)
colour_02 = (0.880722, 0.611041, 0.142051)
colour_03 = (0.560181, 0.691569, 0.194885)
colour_04 = (0.922526, 0.385626, 0.209179)
colour_05 = (0.528488, 0.470624, 0.701351)
colour_07 = (0.772079, 0.431554, 0.102387)
colour_06 = (0.363898, 0.618501, 0.782349)
colour_08 = (1, 0.75, 0)
colour_09 = (0.647624, 0.37816, 0.614037)
mathematica = [colour_01, colour_02, colour_03, colour_04, colour_05, colour_06, colour_07, colour_08, colour_09]
