import pickle
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from utils import NOP, NAME_MAP
from utils import load_dataset, mathematica

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('pgf')

# 1. plot dataset observation and analysis

# utils
def load_raw_dataset(toponame: str, result_dir: str = "./result") -> tuple[np.array, int]:
    result_file = open(f"{result_dir}/{toponame}-solution-optimal.pkl", "rb")
    result = pickle.load(result_file)
    ids = result["id"]
    thetas, times = result["theta"], result["time"]
    sols = result["solution"]
    num_nodes = len(sols[0])

    SP = -2
    NOP = -1
    def transform_action_matrix(mat: np.array, n: int) -> np.array:
        """given a NxN matrix, replace shortest path to a special value"""
        for i in range(n):
            for j in range(n):
                if mat[i][j] == i or mat[i][j] == j:
                    mat[i][j] = SP
        return mat

    dataset = []
    for id, theta, time, sol in zip(ids, thetas, times, sols):
        sol = np.array(sol)
        sol = transform_action_matrix(sol, len(sol))
        # print(id, theta, time, sol.shape)
        dataset.append(sol.flatten())
    dataset = np.array(dataset).T  # shape: [num_flows, num_samples]
    return dataset, num_nodes

def process_raw_dataset(dataset, num_nodes):
    # dataset : [num_flows, num_samples]
    features = []
    for data in dataset:
        feature = [0] * (num_nodes + 2)
        data = data + 2 # NOP -> -1, SP -> -2
        c = Counter(data)
        for k, v in c.items():
            feature[k] = v
        features.append(feature)
    features = np.array(features)
    assert features.shape == (num_nodes * num_nodes, num_nodes + 2)

    # if large, PCA
    if num_nodes > 50:
        pca = PCA(n_components=50, random_state=1024)
        features = pca.fit_transform(features)
    return features

# 1. plot data analysis figure

# 1.1 clustering
def plot_tsne_raw(toponame):
    result_dir = './result/'
    dataset, num_nodes = load_raw_dataset(toponame, result_dir)

    features = process_raw_dataset(dataset, num_nodes)

    # observe per node
    last = 0
    for i in range(num_nodes, num_nodes * num_nodes + 1, num_nodes):
        print(f'node: {last // num_nodes}')
        dataset = []
        flows = []

        for flow in range(num_nodes):
            dataset.append(features[last+flow])
            flows.append(flow)
        dataset = np.array(dataset)

        fig = plt.figure()
        fig.set_figwidth(5)
        fig.set_figheight(4)
        X_embedded = TSNE(
            n_components=2, learning_rate="auto", init="pca", perplexity=5, random_state=1024
        ).fit_transform(dataset)

        d = {'flow': flows, 'x': X_embedded[:, 0], 'y': X_embedded[:, 1]}
        df = pd.DataFrame(d)
        g = sns.scatterplot(data=df, x="x", y="y", alpha=0.9, legend=False, s=100)
        g.set(xticklabels=[])
        g.set(yticklabels=[])
        g.set(xlabel="t-SNE component 1")
        g.set(ylabel="t-SNE component 2")
        plt.savefig(f"./fig/{toponame}/tsne-{toponame}-{last // num_nodes}.pdf", format="pdf", bbox_inches='tight')
        plt.close()
        last = i


# 1.2 plot curve
def plot_gini_and_curve_all_in_one():

    def gini(array):
        """Calculate the Gini coefficient of a numpy array."""
        # based on bottom eq:
        # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
        # from:
        # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
        # All values are treated equally, arrays must be 1d:
        array = array.flatten()
        if np.amin(array) < 0:
            # Values cannot be negative:
            array -= np.amin(array)
        # Values cannot be 0:
        array = array + 0.0000001
        # Values must be sorted:
        array = np.sort(array)
        # Index per array element:
        index = np.arange(1, array.shape[0] + 1)
        # Number of array elements:
        n = array.shape[0]
        # Gini coefficient:
        return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

    def get_padding_freqs(dataset, num_nodes):
        """return each flows' counter, shape [num_flows, num_nodes]  """
        freqs = []
        num_flows = num_nodes * num_nodes
        for flow in range(num_flows):
            c = Counter(dataset[flow].tolist())
            c_exnop = list(filter(lambda x: x[0] != NOP, c.items()))
            if len(c_exnop) < 1:
                continue
            c = sorted(c_exnop, key=lambda x: -x[1])
            f = [v for _, v in c]

            len_rem = num_nodes - len(f)
            if len_rem > 0:
                f.extend([0] * len_rem)

            assert len(f) == num_nodes
            freqs.append(f)
        return freqs

    result_dir = './result/'
    topos = ["Abilene", "nobel", "GEANT", "rf1755", "rf6461", "rf3967", "rf3257", "germany50", "rf1221"]
    avg_df = []
    for toponame in topos:
        dataset, num_nodes = load_raw_dataset(toponame, result_dir)
        avg_df = []
        freqs = get_padding_freqs(dataset, num_nodes)
        ginis = []
        cdfs = []
        max_len = max(map(len, freqs))

        for flow, freq in enumerate(freqs):
            # compute gini
            gini_value = gini(np.array(freq))
            ginis.append(gini_value)

            freq.sort(key=lambda x: -x)
            freq = np.array(freq)
            pdf = freq / sum(freq)
            cdf = np.cumsum(pdf)
            cdf = np.insert(cdf, 0, 0.0)
            cdfs.append(cdf)

            for idx, cdf in enumerate(cdf):
                avg_df.append([toponame, flow, cdf, idx])

        print("mean gini:", np.array(ginis).mean())

        fig = plt.figure()
        fig.set_figwidth(5)
        fig.set_figheight(3.2)
        avg_df = pd.DataFrame(avg_df, columns=["Topology", "flow", "CDF", "Top Node"])
        sns.pointplot(data=avg_df, x="Top Node", y="CDF", errorbar=None, hue="Topology",
                     legend=False)
        sns.lineplot(data=avg_df, x="Top Node", y="CDF", errorbar="pi", hue="Topology",
                     legend=False)

        cdfs = np.array(cdfs).mean(axis=0)
        print(cdfs.shape)
        ax = plt.gca()
        xpoints = (0.0, float(max_len))
        ypoints = (0.0, 1.0)
        print(xpoints, ypoints)
        ax.plot(
            xpoints, ypoints, linestyle="--", color="k", lw=2, scalex=False, scaley=False
        )
        plt.axvline(x=5, ymin=0, ymax=1, color='#EF7579', lw=2)

        integer_locator = matplotlib.ticker.MaxNLocator(integer=True, nbins="auto")
        ax.xaxis.set_major_locator(integer_locator)
        plt.xlim(0, max_len)

        plt.savefig(f"./fig/{toponame}-node-cdf-rev.pdf", format="pdf", bbox_inches='tight')
        plt.close()

# 2. plot SPOP results

# utils
def load_result(toponame, result_dir, method, nc, cd, t):
    tt = f"-t{t}" if t is not None else ""
    # tt = f"-t{t}"
    filename = f"{toponame}-result-{method}-nc-{nc}-cd-{cd}{tt}.pkl"
    file = open(f'{result_dir}/{filename}', 'rb')
    # { "id": ids, "obj": objs, "time": times }
    results = pickle.load(file)
    print(f'f: {filename}', 'mlu:', np.array(results['theta']).mean())
    return results

def load_repetita_result(toponame, result_dir, method, t):
    filename = f"{toponame}-{method}-t{t}.pkl"
    file = open(f'{result_dir}/{filename}', 'rb')
    # { "id": ids, "obj": objs, "time": times }
    results = pickle.load(file)
    return results

# 2.1 plot comparison of baselines
def plot_baselines_comparison():
    data = []
    ratios = []
    topos = ["nobel", "GEANT", "rf1755", "rf6461"]

    def load_tnsm(toponame, result_dir, n_iter=200):
        mlu_file = f'{result_dir}/{toponame}_gravity_1_test.txt'
        time_file = f'{result_dir}/times_{toponame}_gravity_1.txt'

        mlus = []
        lines = open(mlu_file, 'r').readlines()
        for line in lines:
            mlu = float(line.strip())
            mlus.append(mlu)

        times = []
        lines = open(time_file, 'r').readlines()
        for line in lines:
            time = float(line.strip())
            times.append(time)

        result = { 'id': list(range(len(mlus))), 'obj': mlus, 'time': times }
        print(f"{toponame} - tnsm - {n_iter}: ", np.array(mlus).mean())
        return result

    def load_icnp(toponame, result_dir):
        file = f'{result_dir}/{toponame}.csv'
        lines = open(file, 'r').readlines()
        ids = []
        mlus = []
        times = []
        last = 0.0
        for i, line in enumerate(lines):
            ts, id, mlu = line.split(',')
            ts, id, mlu = float(ts), int(id), float(mlu)
            ids.append(id)
            mlus.append(mlu)
            if i == 0:
                times.append(0)
            else:
                times.append(ts - last)
            last = ts
        result = { 'id': ids, 'obj': mlus, 'time': times }
        print(f"{toponame} - icnp: ", np.array(mlus).mean())
        return result

    def load_repetita(toponame, method, t):
        result_dir = './result/'
        result = load_repetita_result(toponame, result_dir, method, t)
        return result

    def load_ours(toponame, method, nc, cd, t):
        result_dir = './result0.1/'
        result = load_result(toponame, result_dir, method, nc, cd, t)
        return result

    def split_dataset(num_tm, seed=1024, ratio=0.7):
        """return idxes of splitted data set"""
        idxes = np.arange(num_tm)
        np_state = np.random.RandomState(seed)
        np_state.shuffle(idxes)

        len_idxes = len(idxes)
        trainsize = int(ratio*len_idxes)
        trainset, testset = idxes[:trainsize], idxes[trainsize:]
        return trainset, testset

    for toponame in topos:
        icnp = load_icnp(toponame, result_dir='./result.icnp')
        tnsm = load_tnsm(toponame, result_dir='./result.tnsm')
        defo = load_repetita(toponame, "defoCP", 2)
        defo['obj'] = defo['theta']
        srls = load_repetita(toponame, "SRLS", 2)
        srls['obj'] = srls['theta']
        if "rf" in toponame:
            ours = load_ours(toponame, "rsp", 5, 5, 2)
        else:
            ours = load_ours(toponame, "rs", None, 5, 2)
        ours['obj'] = ours['theta']
        opt = load_result(toponame, "./result", "opt", None, None, None)
        opt['obj'] = opt['theta']
        stretch = load_result(toponame, "./result", "str", None, 5, None)
        stretch['obj'] = stretch['theta']

        opt_map = dict()
        for (id, reward, time) in zip(opt['id'], opt['obj'], opt['time']):
            opt_map[id] = reward

        names = ["tnsm", "icnp", "str", "defo", "srls", "ours", "opt"]
        datas = [tnsm, icnp, stretch, defo, srls, ours, opt]

        for name, result in zip(names, datas):
            t = np.array(result.get('time', [2])).mean()
            m = np.array(result.get('obj')).mean()
            print(f'{toponame} {name} time: {t} mlu: {m}')

        for method, result in zip(names, datas):
            idxes, rewards, times = result['id'], result['obj'], result.get('time', [2] * len(result['id']))

            if method == "tnsm" or method == "icnp":
                num_tm = 2016 if toponame == "GEANT" else 288
                _, idxes = split_dataset(num_tm=num_tm)

            for (id, reward, time) in zip(idxes, rewards, times):
                name = NAME_MAP[method]
                data.append([name, reward, time, toponame])
                ratios.append([name, reward / opt_map[id], time, toponame])

    df = pd.DataFrame(
        data, columns=['Method', 'Max Link Utilization', 'Time', 'Topology'])

    rdf = pd.DataFrame(
        ratios, columns=['Method', 'Performance Ratio', 'Time', 'Topology'])

    df.loc[df.Topology == 'nobel', 'Topology'] = 'NOBEL'
    rdf.loc[rdf.Topology == 'nobel', 'Topology'] = 'NOBEL'

    def plot_bar(df):
        # all-in-one
        fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharey=False, squeeze=True)
        plt.subplots_adjust(wspace=0.3)
        for i, topo in enumerate(df.Topology.unique()):
            row = i // 2
            col = i % 2

            _df = df[df.Topology == topo]

            # fig = plt.figure()
            # fig.set_figwidth(10)
            # fig.set_figheight(5)
            # sns.boxplot(data=df, x="Toponame", y="Max Link Utilization",
            #               hue="Method")
            # sns.barplot(data=df, x="Toponame")
            # matplotlib.style.use("default")
            hue_order = ["MARL-GNN", "DeepLS", "STR", "DEFO", "SRLS", "Ours", "ILP"]

            colors = mathematica.copy()
            sns.set_theme(style='whitegrid', font_scale=1.2)
            aaa = sns.barplot(ax=ax[row][col], data=_df, x="Topology", y="Max Link Utilization", hue="Method",
                        palette=colors,
                        hue_order=hue_order,
                        # estimator=lambda x: x.mean().round(3),
                        legend='auto',
                        errorbar="sd", capsize=.3)
            # plt.ylim(0.2, 0.8)
            # for i in aaa.containers:
            #     aaa.bar_label(i)


            opt_mean = _df[_df.Method == "LP"]["Max Link Utilization"].mean()
            mean = _df[_df.Method == "Ours"]["Max Link Utilization"].mean()
            print(f"{topo} mean ratio: {mean/opt_mean}")

            our_idx = 5

            # num_locations = len(df.Method.unique())
            hatches = ['.', '/', '-', '\\', '+', '.', 'o', 'O']
            # l = ax[i].legend(loc='best', ncol=1, fancybox=True, shadow=False)
            l = ax[row][col].get_legend()
            for j, bar in enumerate(ax[row][col].patches[:7]):
                # print(j, bar)
                if j == our_idx:
                    continue
                bar.set_hatch(hatches[j])
            for j, bar in enumerate(l.get_patches()):
                if j == our_idx:
                    continue
                bar.set_hatch(hatches[j] * 2)

            ax[row][col]._remove_legend(l)
            ax[row][col].set(xlabel='')

        handles, labels = ax[0][0].get_legend_handles_labels()
        l = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=7, title=None, frameon=False, fontsize="small")
        hatches = ['.', '/', '-', '\\', '+', '.', 'o', 'O']
        for j, bar in enumerate(l.get_patches()):
            if j == 5:
                continue
            bar.set_hatch(hatches[j] * 2)

        plt.tight_layout()
        plt.savefig(
            f"./fig/performance.pdf", format="pdf", bbox_inches='tight')
        plt.close()

    def plot_ratio(df):
        # all-in-one
        fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharey=False, squeeze=True)
        plt.subplots_adjust(wspace=0.3)
        for i, topo in enumerate(df.Topology.unique()):
            row = i // 2
            col = i % 2

            _df = df[df.Topology == topo]

            hue_order = ["MARL-GNN", "DeepLS", "STR", "DEFO", "SRLS", "Ours"]

            colors = mathematica.copy()
            sns.set_theme(style='whitegrid', font_scale=1.2)
            # plt.ylim(0.2, 0.8)

            ls = ['--', '-.', ':', (0, (6, 2)), (0, (2, 1)), '-', '-']
            sns.ecdfplot(ax=ax[row][col], data=_df, x="Performance Ratio", hue="Method", lw=2, hue_order=hue_order)
            for lines, linestyle in zip(ax[row][col].lines[::-1], ls):
                # print(lines, linestyle)
                lines.set_linestyle(linestyle)

        handles, labels = ax[0][0].get_legend_handles_labels()
        l = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=7, title=None, frameon=False, fontsize="small")

        plt.tight_layout()
        plt.savefig(
            f"./fig/performance-ratio.pdf", format="pdf", bbox_inches='tight')
        plt.close()


    plot_bar(df)
    plot_ratio(rdf)


# 2.2 plot impact of number of clusters
def plot_cluster_number():
    data = []
    topos = ["GEANT", "nobel", "rf1755", "rf6461"]

    def load_ours(toponame, result_dir, method, nc, cd, t):
        result = load_result(toponame, result_dir, method, nc, cd, t)
        return result

    ratio = 0.3
    for toponame in topos:
        data = []
        for nc in [3, 4, 5, 6, 7, 8, None]:
            result_dir = f'./result{str(ratio)}/'
            method = "rs" if nc is None else "rsp"
            result = load_ours(toponame, result_dir, method=method, nc=nc, cd=5, t=2)
            result['obj'] = result['theta']

            idxes, rewards, times = result['id'], result['obj'], result.get('time', [2] * len(result['id']))
            for (_, reward, time) in zip(idxes, rewards, times):
                if nc is None:
                    data.append([reward, time, toponame, 'w/o'])
                else:
                    data.append([reward, time, toponame, str(nc)])

        df = pd.DataFrame(
            data, columns=['Max Link Utilization', 'Time', 'Topology', 'Number of clusters'])

        fig = plt.figure()
        fig.set_figwidth(5)
        fig.set_figheight(3.2)
        sns.set_theme(style='whitegrid', font_scale=1.2)
        sns.boxplot(data=df, x="Number of clusters", y="Max Link Utilization",
                    color = "#7DAEE0",
                    fill=True)

        plt.tight_layout()
        plt.savefig(
            f"./fig/{toponame}-nc-ratio.pdf", format="pdf", bbox_inches='tight')
        plt.close()

# 2.3 plot impact of number of training set
def plot_training_set_number():
    data = []
    topos = ["GEANT", "nobel", "rf1755", "rf6461"]

    def load_ours(toponame, result_dir, method, nc, cd, t):
        result = load_result(toponame, result_dir, method, nc, cd, t)
        return result

    for toponame in topos:
        data = []
        for ratio in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            result_dir = f'./result{str(ratio)}/'
            result = load_ours(toponame, result_dir, method="rsp", nc=5, cd=5, t=2)
            result['obj'] = result['theta']

            idxes, rewards, times = result['id'], result['obj'], result.get('time', [2] * len(result['id']))
            for (_, reward, time) in zip(idxes, rewards, times):
                data.append([reward, time, toponame, str(ratio)])

        df = pd.DataFrame(
            data, columns=['Max Link Utilization', 'Time', 'Topology', 'Ratio'])

        fig = plt.figure()
        fig.set_figwidth(5)
        fig.set_figheight(3.2)
        sns.set_theme(style='whitegrid', font_scale=1.2)
        sns.boxplot(data=df, x="Ratio", y="Max Link Utilization",
                    color="#ffb77f",
                    fill=True)

        plt.tight_layout()
        plt.savefig(
            f"./fig/{toponame}-dataset-ratio.pdf", format="pdf", bbox_inches='tight')
        plt.close()


# 2.4 plot impact of link failures
def plot_linkfail():
    result_dir = "./result.linkfail/"
    topos = ["nobel", "GEANT", "rf1755", "rf6461"]

    def plot_linkfail_topo(toponame, result_dir):
        """plot linkfail experiments"""

        valid_method = ['sns', 'fan', 'snsc']

        def load_linkfail_result(toponame, result_dir, method, nc, cd, t, num_linkfail):
            tt = f"-t{t}" if t is not None else ""
            linkfail = f"-linkfail-{num_linkfail}" if num_linkfail > 0 else ""
            filename = f"{toponame}-result-{method}-nc-{nc}-cd-{cd}{tt}{linkfail}.pkl"
            file = open(f'{result_dir}/{filename}', 'rb')
            # [{ "id": ids, "obj": objs, "time": times }]
            results = pickle.load(file)
            # print(f'f: {filename}', 'mlu:', np.array(results['theta']).mean())
            theta = []
            for result in results:
                theta.extend(result.get('obj', result.get('theta')))
            print(f'f: {filename}', 'mlu:', np.array(theta).mean())
            return results

        NUM_LINKFAILS = [1, 2, 3, 5]

        data = []

        # 1. load no linkfail data
        no_fail_result_dir = "./result/"
        opt_dict = {}
        opt_result = load_result(toponame, no_fail_result_dir, "opt", nc=None, cd=None, t=None)
        idxes, rewards, times = opt_result['id'], opt_result['theta'], opt_result['time']
        for (id, reward, time) in zip(idxes, rewards, times):
            opt_dict[id] = reward

        ## 1.1 get needed idxes
        idx_set = set()
        if "rf" in toponame:
            results = load_linkfail_result(toponame, result_dir=result_dir, method="rsp", nc=5, cd=5, t=2, num_linkfail=1)
        else:
            results = load_linkfail_result(toponame, result_dir=result_dir, method="rs", nc=None, cd=5, t=2, num_linkfail=1)
        for result in results:
            idxes = result['id']
            idx_set |= set(idxes)

        ## 1.2 get result of needed idxes
        spop_dir = "./result0.1/"
        if "rf" in toponame:
            result = load_result(toponame, result_dir=spop_dir, method="rsp", nc=5, cd=5, t=2)
        else:
            result = load_result(toponame, result_dir=spop_dir, method="rs", nc=None, cd=5, t=2)
        idxes, rewards, times = result['id'], result['theta'], result['time']

        for (id, reward, time) in zip(idxes, rewards, times):
            if id in idx_set:
                opt_reward = opt_dict[id]
                data.append(
                    [NAME_MAP["ours"], reward / opt_reward, time, str(0)])

        # load linkfail data
        for num_linkfail in NUM_LINKFAILS:
            if "rf" in toponame:
                results = load_linkfail_result(toponame, result_dir=result_dir, method="rsp", nc=5, cd=5, t=2, num_linkfail=num_linkfail)
            else:
                results = load_linkfail_result(toponame, result_dir=result_dir, method="rs", nc=None, cd=5, t=2, num_linkfail=num_linkfail)

            # if toponame != "rf6461":
            opt_results = load_linkfail_result(toponame, result_dir=result_dir, method="opt",
                                            nc=None, cd=None, t=10000, num_linkfail=num_linkfail)
            for result, opt_result in zip(results, opt_results):
                opt_dict = {}
                idxes, rewards, times = opt_result['id'], opt_result.get('theta', opt_result.get('obj')), opt_result['time']
                for (id, reward, time) in zip(idxes, rewards, times):
                    opt_dict[id] = reward

                idxes, rewards, times = result['id'], result['theta'], result['time']
                for (id, reward, time) in zip(idxes, rewards, times):
                    if id not in opt_dict:
                        continue
                    opt_reward = opt_dict[id]
                    data.append(
                        [NAME_MAP["ours"], reward / opt_reward, time, str(num_linkfail)])
        df = pd.DataFrame(
            data, columns=['Method', 'Performance Ratio', 'Time', 'Link Failure(s)'])

        fig = plt.figure()
        fig.set_figwidth(5)
        fig.set_figheight(3.2)

        g = sns.pointplot(data=df, x="Link Failure(s)", y="Performance Ratio", hue="Method",
                        errorbar="pi", capsize=0.2, legend=False)

        for l in g.lines:
            plt.setp(l,linewidth=1.5)

        plt.tight_layout()
        plt.savefig(f"./fig/linkfail-{toponame}.pdf", format="pdf", bbox_inches='tight')
        plt.close()


    for toponame in topos:
        plot_linkfail_topo(toponame, result_dir)

def plot_linkfail_all_in_one():
    result_dir = "./result.linkfail/"
    topos = ["nobel", "GEANT", "rf1755", "rf6461"]

    def load_linkfail_result(toponame, result_dir, method, nc, cd, t, num_linkfail):
        tt = f"-t{t}" if t is not None else ""
        linkfail = f"-linkfail-{num_linkfail}" if num_linkfail > 0 else ""
        filename = f"{toponame}-result-{method}-nc-{nc}-cd-{cd}{tt}{linkfail}.pkl"
        file = open(f'{result_dir}/{filename}', 'rb')
        # [{ "id": ids, "obj": objs, "time": times }]
        results = pickle.load(file)
        # print(f'f: {filename}', 'mlu:', np.array(results['theta']).mean())
        theta = []
        for result in results:
            theta.extend(result.get('obj', result.get('theta')))
        print(f'f: {filename}', 'mlu:', np.array(theta).mean())
        return results

    NUM_LINKFAILS = [1, 2, 3, 5]

    data = []

    for toponame in topos:

        # 1. load no linkfail data
        no_fail_result_dir = "./result/"
        opt_dict = {}
        opt_result = load_result(toponame, no_fail_result_dir, "opt", nc=None, cd=None, t=None)
        idxes, rewards, times = opt_result['id'], opt_result['theta'], opt_result['time']
        for (id, reward, time) in zip(idxes, rewards, times):
            opt_dict[id] = reward

        ## 1.1 get needed idxes
        idx_set = set()
        if "rf" in toponame:
            results = load_linkfail_result(toponame, result_dir=result_dir, method="rsp", nc=5, cd=5, t=2, num_linkfail=1)
        else:
            results = load_linkfail_result(toponame, result_dir=result_dir, method="rs", nc=None, cd=5, t=2, num_linkfail=1)
        for result in results:
            idxes = result['id']
            idx_set |= set(idxes)

        ## 1.2 get result of needed idxes
        spop_dir = "./result0.1/"
        if "rf" in toponame:
            result = load_result(toponame, result_dir=spop_dir, method="rsp", nc=5, cd=5, t=2)
        else:
            result = load_result(toponame, result_dir=spop_dir, method="rs", nc=None, cd=5, t=2)
        idxes, rewards, times = result['id'], result['theta'], result['time']

        for (id, reward, time) in zip(idxes, rewards, times):
            if id in idx_set:
                opt_reward = opt_dict[id]
                data.append(
                    [NAME_MAP["ours"], reward / opt_reward, time, str(0), toponame])

        # load linkfail data
        for num_linkfail in NUM_LINKFAILS:
            if "rf" in toponame:
                results = load_linkfail_result(toponame, result_dir=result_dir, method="rsp", nc=5, cd=5, t=2, num_linkfail=num_linkfail)
            else:
                results = load_linkfail_result(toponame, result_dir=result_dir, method="rs", nc=None, cd=5, t=2, num_linkfail=num_linkfail)

            # if toponame != "rf6461":
            opt_results = load_linkfail_result(toponame, result_dir=result_dir, method="opt",
                                            nc=None, cd=None, t=10000, num_linkfail=num_linkfail)
            for result, opt_result in zip(results, opt_results):
                if toponame != "rf6461":
                    opt_dict = {}
                    idxes, rewards, times = opt_result['id'], opt_result.get('theta', opt_result.get('obj')), opt_result['time']
                    for (id, reward, time) in zip(idxes, rewards, times):
                        opt_dict[id] = reward

                idxes, rewards, times = result['id'], result['theta'], result['time']
                for (id, reward, time) in zip(idxes, rewards, times):
                    if id not in opt_dict:
                        continue
                    opt_reward = opt_dict[id]
                    data.append(
                        [NAME_MAP["ours"], reward / opt_reward, time, str(num_linkfail), toponame])

            df = pd.DataFrame(
                data, columns=['Method', 'Performance Ratio', 'Time', 'Link Failure(s)', 'Topology'])

            df.loc[df.Topology == 'nobel', 'Topology'] = 'NOBEL'

    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(3.5)

    g = sns.pointplot(data=df, x="Link Failure(s)", y="Performance Ratio", hue="Topology",
                      markers=["o", "s", "D", "x"],
                    #   linestyles=["-", "--", ":", "-."],
                      dodge=0.5, errorbar="pi", capsize=0.2)

    for l in g.lines:
        plt.setp(l,linewidth=1.5)

    plt.tight_layout()
    plt.savefig(f"./fig/linkfail-all-in-one.pdf", format="pdf", bbox_inches='tight')
    plt.close()


# 2.5 plot result on uniform TMs
def plot_performance_ratio_over_time():
    topos = ["nobel", "GEANT", "rf1755", "rf6461"]

    for toponame in topos:
        data = []

        # load_real
        opt_dict = {}
        opt_result = load_result(toponame, "./result", "opt", nc=None, cd=None, t=None)
        idxes, rewards, times = opt_result['id'], opt_result['theta'], opt_result['time']
        for (id, reward, time) in zip(idxes, rewards, times):
            opt_dict[id] = reward

        spop_dir = "./result0.1/"
        if "rf" in toponame:
            result = load_result(toponame, result_dir=spop_dir, method="rsp", nc=5, cd=5, t=2)
        else:
            result = load_result(toponame, result_dir=spop_dir, method="rs", nc=None, cd=5, t=2)
        idxes, rewards, times = result['id'], result['theta'], result['time']

        i = 0
        for (id, reward, time) in zip(idxes, rewards, times):
            opt_reward = opt_dict[id]
            data.append(
                [i, toponame, NAME_MAP["ours"], reward / opt_reward, time, "Test"])
            i += 1

        # load_uniform
        opt_dict = {}
        opt_result = load_result(toponame, "./result.uniform", "opt", nc=None, cd=None, t=None)
        idxes, rewards, times = opt_result['id'], opt_result['theta'], opt_result['time']
        for (id, reward, time) in zip(idxes, rewards, times):
            opt_dict[id] = reward

        spop_dir = "./result.uniform/"
        if "rf" in toponame:
            result = load_result(toponame, result_dir=spop_dir, method="rsp", nc=5, cd=5, t=2)
        else:
            result = load_result(toponame, result_dir=spop_dir, method="rs", nc=None, cd=5, t=2)
        idxes, rewards, times = result['id'], result['theta'], result['time']

        for (id, reward, time) in zip(idxes, rewards, times):
            opt_reward = opt_dict[id]
            data.append(
                [i, toponame, NAME_MAP["ours"], reward / opt_reward, time, "Uniform"])
            i += 1

        # load_graivty
        opt_dict = {}
        opt_result = load_result(toponame, "./result.gravity", "opt", nc=None, cd=None, t=None)
        idxes, rewards, times = opt_result['id'], opt_result['theta'], opt_result['time']
        for (id, reward, time) in zip(idxes, rewards, times):
            opt_dict[id] = reward

        spop_dir = "./result.gravity/"
        if "rf" in toponame:
            result = load_result(toponame, result_dir=spop_dir, method="rsp", nc=5, cd=5, t=2)
        else:
            result = load_result(toponame, result_dir=spop_dir, method="rsp", nc=None, cd=5, t=2)
        idxes, rewards, times = result['id'], result['theta'], result['time']

        for (id, reward, time) in zip(idxes, rewards, times):
            opt_reward = opt_dict[id]
            data.append(
                [i, toponame, NAME_MAP["ours"], reward / opt_reward, time, "Gravity"])
            i += 1

        df = pd.DataFrame(
            data, columns=['TM', 'Topology', 'Method', 'Performance Ratio', 'Time', 'TM Type'])

        # plot pr over time
        fig = plt.figure()
        fig.set_figwidth(5)
        fig.set_figheight(3.2)
        sns.set_theme(style='whitegrid', font_scale=1.2)

        sns.lineplot(data=df, x="TM", y="Performance Ratio",
                     hue="TM Type",
                     style="TM Type",
                     dashes=True)

        plt.axvline(x=i//3, color='black', linewidth=1.0, linestyle='--')
        plt.axvline(x=i//3*2, color='black', linewidth=1.0, linestyle='--')
        plt.savefig(f"./fig/{toponame}-over-time.pdf", format="pdf", bbox_inches='tight')

        # plot pr CDF
        fig, ax = plt.subplots(1, 1)
        fig.set_figwidth(4)
        fig.set_figheight(2.3)
        sns.set_theme(style='whitegrid', font_scale=1.2)

        palette = ["#ef767a", "#456990", "#48c0aa"]
        sns.ecdfplot(ax=ax, data=df, x="Performance Ratio", hue="TM Type", lw=2, hue_order=["Test", "Uniform", "Gravity"], palette=palette)
        fig.gca().set(ylabel="CDF")

        ls = ['-', '-.', '--']
        for line, linestyle in zip(ax.lines[::-1], ls):
            line.set_linestyle(linestyle)
        lg = ax.get_legend()
        for line, linestyle in zip(lg.get_lines(), ls):
            line.set_linestyle(linestyle)


        plt.savefig(f"./fig/{toponame}-cdf.pdf", format="pdf", bbox_inches='tight')



# 2.6 plot observation of clustering
def plot_tsne_with_clusters(toponame, m, per_node = False):

    def load_clusters(nc_dir, toponame, m: int, per_node: bool):
        if per_node:
            file = open(f"{nc_dir}/{toponame}-rsp-nc-{m}.pkl", "rb")
        else:
            file = open(f"{nc_dir}/{toponame}-rs-nc-{m}.pkl", "rb")

        clusters = pickle.load(file)
        return clusters


    nc_dir = './nc0.1/'
    result_dir = './result/'
    dataset, num_nodes = load_raw_dataset(toponame, result_dir)
    clusters = load_clusters(nc_dir, toponame, m, per_node)

    features = process_raw_dataset(dataset, num_nodes)

    # observe per node
    last = 0
    for i in range(num_nodes, num_nodes * num_nodes + 1, num_nodes):
        print(f'node: {last // num_nodes}')
        dataset = []
        flows = []
        node_clusters = []

        for flow in range(num_nodes):
            dataset.append(features[last+flow])
            flows.append(flow)
            node_clusters.append(clusters[last + flow])
        dataset = np.array(dataset)

        fig = plt.figure()
        fig.set_figwidth(5)
        fig.set_figheight(4)
        X_embedded = TSNE(
            n_components=2, learning_rate="auto", init="pca", perplexity=5, random_state=1024
        ).fit_transform(dataset)

        d = {'flow': flows, 'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'c': node_clusters}
        df = pd.DataFrame(d)
        g = sns.scatterplot(data=df, x="x", y="y", hue="c", style="c", alpha=0.9,
                            palette="deep", legend=False, s=100)

        g.set(xticklabels=[])
        g.set(yticklabels=[])
        g.set(xlabel="t-SNE component 1")
        g.set(ylabel="t-SNE component 2")
        plt.savefig(f"./fig/{toponame}/tsne-{toponame}-{last // num_nodes}-cluster.pdf", format="pdf", bbox_inches='tight')
        plt.close()
        last = i

# 2.7 plot comparison between no-traffic-splitting (binary variables) and traffic-splitting (continous variables)
def plot_traffic_splitting_comparison():
    topos = ["nobel", "GEANT", "rf1755", "rf6461"]

    def load_ilp(toponame, method, nc, cd, t):
        result_dir = './result0.1/'
        result = load_result(toponame, result_dir, method, nc, cd, t)
        return result

    def load_lp(toponame, method, nc, cd, t):
        result_dir = './result.lp/'
        result = load_result(toponame, result_dir, method, nc, cd, t)
        return result

    data = []
    for toponame in topos:
        if "rf" in toponame:
            ilp = load_ilp(toponame, "rsp", 5, 5, 2)
        else:
            ilp = load_ilp(toponame, "rs", None, 5, 2)

        opt_ilp = load_result(toponame, "./result", "opt", None, None, None)

        if "rf" in toponame:
            lp = load_lp(toponame, "rsp", 5, 5, 2)
        else:
            lp = load_lp(toponame, "rsp", None, 5, 2)

        opt_lp = load_result(toponame, "./result.lp", "opt", None, None, None)

        datasets = [opt_ilp, ilp, opt_lp, lp]
        names = ["ILP", "ILP + RS4SR", "LP", "LP + RS4SR"]
        for name, result in zip(names, datasets):
            idxes, times, rewards = result['id'], result['time'], result['theta']
            for (id, reward, time) in zip(idxes, rewards, times):
                data.append([name, reward, time, toponame])

    df = pd.DataFrame(data, columns=["Method", "Max Link Utilization", "Time", "Topology"])
    df.loc[df.Topology == 'nobel', 'Topology'] = 'NOBEL'

    fig = plt.figure()

    palette = ["#f6b3ac", "#f47f72", "#b3d1e7", "#7fb2d5"]

    ax = sns.boxplot(data=df, x="Topology", y="Max Link Utilization", hue="Method",
                palette = palette,
                fill=True)

    hatches = ['', '', '', '', '//', '//', '//', '//'] * 2
    patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
    for patch, hatch in zip(patches, hatches):
        patch.set_hatch(hatch)

    hatches = ['', '//', '', '//']
    l = ax.legend()
    for lp, hatch in zip(l.get_patches(), hatches):
        lp.set_hatch(hatch)
        lp.set_edgecolor('k')

    fig.set_figwidth(8)
    fig.set_figheight(4)

    sns.set_theme(style='whitegrid', font_scale=1.2)

    plt.savefig(f"./fig/compare-split.pdf", format="pdf", bbox_inches='tight')



def main():
    # tsne
    # custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    # sns.set_theme(style="white", rc=custom_params, font_scale=1.3)
    # plot_tsne_raw("Abilene")
    # # plot_tsne_raw("germany50")
    # plot_tsne_raw("nobel")
    # plot_tsne_raw("GEANT")
    # plot_tsne_raw("rf1221")
    # plot_tsne_raw("rf1755")
    # plot_tsne_raw("rf6461")
    # plot_tsne_raw("rf3257")
    # plot_tsne_raw("rf3967")
    # plot_tsne_with_clusters("nobel", m=5, per_node=True)
    # plot_tsne_with_clusters("GEANT", m=5, per_node=True)
    # plot_tsne_with_clusters("rf1755", m=5, per_node=True)
    # plot_tsne_with_clusters("rf6461", m=5, per_node=True)

    # gini
    # sns.set_theme(style='ticks', font_scale=1.1)
    # plot_gini_and_curve_all_in_one()

    # sns.set_theme(style='whitegrid', font_scale=1.1)
    # plot_performance_ratio_over_time()

    # sns.set_theme(style='whitegrid', font_scale=1.1)
    # plot_traffic_splitting_comparison()

    sns.set_theme(style='whitegrid', font_scale=1.3)
    # plot_training_set_number()
    # plot_cluster_number()
    # plot_baselines_comparison()

    # sns.set_theme(style='whitegrid', font_scale=1.1)
    # plot_linkfail_all_in_one()


main()
