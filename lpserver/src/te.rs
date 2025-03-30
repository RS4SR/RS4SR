use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use petgraph::algo::{all_shortest_paths, toposort, FloatMeasure};
use petgraph::graphmap::DiGraphMap;
use petgraph::visit::{EdgeRef, NodeIndexable};
use rayon::prelude::*;

use crate::config::{get_candidate, Candidates, Config, ModelType, Solution, TLMAP};
use crate::lp::ilp;
use crate::lp::lp;

#[derive(Debug, Default, Eq, PartialOrd, Ord, Copy, Clone)]
pub struct EdgeAttr {
    weight: u32,
    cap: u32,
}

impl EdgeAttr {
    pub fn new(weight: u32, cap: u32) -> EdgeAttr {
        EdgeAttr { weight, cap }
    }
}

impl PartialEq for EdgeAttr {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl std::ops::Add for EdgeAttr {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            weight: self.weight + other.weight,
            cap: self.cap,
        }
    }
}

impl FloatMeasure for EdgeAttr {
    fn zero() -> Self {
        EdgeAttr { cap: 0, weight: 0 }
    }

    fn infinite() -> Self {
        EdgeAttr {
            cap: 0,
            weight: 0x3f3f3f3f,
        }
    }
}

pub type TM = Vec<Vec<f64>>;
type FMap = HashMap<(usize, usize), HashMap<(usize, usize), f64>>;

pub fn load_tm_list(path: impl AsRef<Path>) -> Vec<TM> {
    let content = std::fs::read(path).unwrap();
    serde_pickle::from_slice(&content, Default::default()).unwrap()
}

pub fn load_topology(path: impl AsRef<Path>) -> DiGraphMap<usize, EdgeAttr> {
    let mut graph = DiGraphMap::new();
    let content: String = std::fs::read_to_string(path).unwrap();
    let node_count = content
        .lines()
        .flat_map(|line| {
            line.trim()
                .splitn(4, ' ')
                .take(2)
                .map(|x| x.parse::<usize>().unwrap())
        })
        .max()
        .unwrap();
    for i in 0..node_count {
        graph.add_node(i);
    }
    for line in content.lines() {
        match line.trim().splitn(5, ' ').collect::<Vec<_>>()[..] {
            [src, dst, weight, cap, ..] => {
                let src: usize = src.parse().unwrap();
                let dst: usize = dst.parse().unwrap();
                let weight: u32 = weight.parse().unwrap();
                let cap: u32 = cap.parse().unwrap();
                let attr = EdgeAttr::new(weight, cap);
                graph.add_edge(src, dst, attr);
            }
            _ => panic!("Invalid input file"),
        }
    }
    // println!("nodes: {} edges: {}", graph.node_count(), graph.edge_count());
    graph
}

/// save data related to network topology
#[derive(Debug)]
pub struct TESolver {
    cfg: Config,
    graph: DiGraphMap<usize, EdgeAttr>,
    pub f_data: FMap,
    edges: Vec<(usize, usize)>,
    caps: Vec<f64>,
    pub global_ratios: Option<Vec<Vec<(usize, f64, usize)>>>, // size f x e x 3, each flow's cand ratio on each each
    pub pool: rayon::ThreadPool,
    pub ratio_compute_time: f64, // unit: seconds
    nc: Option<Vec<usize>>,      // size f x 1, record each flow(f = n x n)'s cluster id
    removed_nodes: Vec<usize>,
}

impl TESolver {
    pub fn new(cfg: Config) -> TESolver {
        let graph = load_topology(cfg.topo_path());
        // create a rayon thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(32)
            .build()
            .unwrap();
        let mut solver = TESolver {
            cfg,
            graph,
            f_data: HashMap::new(),
            edges: Vec::new(),
            caps: Vec::new(),
            global_ratios: None,
            pool,
            ratio_compute_time: 0.0,
            nc: None,
            removed_nodes: Vec::new(),
        };
        solver.remove_edges_randomly().unwrap();
        solver.remove_nodes_randomly().unwrap();
        solver.add_edges_randomly().unwrap();
        solver.precompute_graph_info(); // compute edge, cap info related to topology
        solver.precompute_f(); // compute f,g function related to 2-SR
        solver.precompute_nc(); // compute nc by reading result from file (if needed)
        let now = Instant::now();
        solver.precompute_ratios(); // compute global ratios (constant during running)
        solver.ratio_compute_time = now.elapsed().as_secs_f64();
        solver
    }

    /// receive failed links from client reqeust
    fn remove_edges_randomly(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.cfg.num_linkfail <= 0 {
            // do nothing
            return Ok(());
        }

        let ctx = zmq::Context::new();
        let pull_sock = ctx.socket(zmq::PULL)?;
        let id = self.cfg.id;
        pull_sock.connect(&format!("ipc:///tmp/link_failures{id}"))?;

        let msg = pull_sock.recv_bytes(0)?;
        let failures: Vec<(usize, usize)> = serde_pickle::from_slice(&msg, Default::default())?;
        assert_eq!(self.cfg.num_linkfail, failures.len());
        for (s, t) in failures.into_iter() {
            self.graph.remove_edge(s, t);
        }
        Ok(())
    }

    /// receive removed nodes from client reqeust
    fn remove_nodes_randomly(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.cfg.num_nodermv <= 0 {
            // do nothing
            return Ok(());
        }

        let ctx = zmq::Context::new();
        let pull_sock = ctx.socket(zmq::PULL)?;
        let id = self.cfg.id;
        pull_sock.connect(&format!("ipc:///tmp/node_remove{id}"))?;

        let msg = pull_sock.recv_bytes(0)?;
        let failures: Vec<usize> = serde_pickle::from_slice(&msg, Default::default())?;
        assert_eq!(self.cfg.num_nodermv, failures.len());
        for &n in failures.iter() {
            let mut edges: Vec<(usize, usize)> = vec![];
            for  (s, t, _ ) in self.graph.all_edges() {
                if s == n || t == n {
                    edges.push((s, t));
                }
            }
            for (s, t) in edges {
                self.graph.remove_edge(s, t);
            }
        }
        self.removed_nodes = failures;
        Ok(())
    }

    /// receive added links from client reqeust
    fn add_edges_randomly(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.cfg.num_linkadd <= 0 {
            // do nothing
            return Ok(());
        }

        let ctx = zmq::Context::new();
        let pull_sock = ctx.socket(zmq::PULL)?;
        let id = self.cfg.id;
        pull_sock.connect(&format!("ipc:///tmp/link_adds{id}"))?;

        let msg = pull_sock.recv_bytes(0)?;
        // added links (src, dst, weight, capacity)
        let addeds: Vec<(usize, usize, usize, usize)> =
            serde_pickle::from_slice(&msg, Default::default())?;
        assert_eq!(self.cfg.num_linkadd, addeds.len());
        for (s, t, w, c) in addeds.into_iter() {
            let attr = EdgeAttr::new(w as u32, c as u32);
            self.graph.add_edge(s, t, attr);
            self.graph.add_edge(t, s, attr);
        }
        Ok(())
    }

    fn compute_ecmp_link_frac(
        &self,
        i: usize,
        j: usize,
        load: f64,
    ) -> HashMap<(usize, usize), f64> {
        let idx = |i| NodeIndexable::from_index(&self.graph, i);
        let ix = |i| self.graph.to_index(i);
        let paths = all_shortest_paths(&self.graph, idx(i), idx(j)).unwrap();
        // println!("{i} -> {j}: {:?}", paths);
        // build DAG
        let mut dag = DiGraphMap::new();
        let mut node_succ = HashMap::<usize, HashSet<usize>>::new();
        let mut node_load = HashMap::<usize, f64>::new();
        for p in paths {
            for (s, t) in p.iter().zip(p.iter().skip(1)) {
                let set = node_succ.entry(*s).or_default();
                set.insert(*t);
                dag.add_edge(*s, *t, 0.0);
            }
        }
        // compute fractions
        node_load.entry(idx(i)).or_insert(load);
        for node in toposort(&dag, None).unwrap().into_iter() {
            let nexthops = node_succ.get(&node);
            if nexthops.is_none() {
                continue;
            }
            let nexthops = nexthops.unwrap();
            let next_load = node_load[&node] / nexthops.len() as f64;
            for nexthop in nexthops {
                let w = dag.edge_weight_mut(node, *nexthop).unwrap();
                *w += next_load;
                *node_load.entry(*nexthop).or_insert(0.0) += next_load;
            }
        }
        let mut ans = HashMap::new();
        for e in dag.all_edges() {
            let (s, t) = (ix(e.source()), ix(e.target()));
            ans.insert((s, t), *e.weight());
        }
        ans
    }

    fn precompute_f(&mut self) {
        let node_count = self.graph.node_count();
        for i in 0..node_count {
            for j in 0..node_count {
                if i != j {
                    let ans = self.compute_ecmp_link_frac(i, j, 1.0);
                    self.f_data.insert((i, j), ans);
                }
            }
        }
    }

    fn precompute_graph_info(&mut self) {
        for (s, t, attr) in self.graph.all_edges() {
            self.edges.push((s, t));
            self.caps.push(attr.cap as f64);
        }
    }

    fn precompute_nc(&mut self) {
        if self.cfg.num_nc.is_some() {
            let path = self.cfg.nc_path();
            let content = std::fs::read(path).unwrap();
            self.nc = serde_pickle::from_slice(&content, Default::default()).unwrap();
        }
    }

    pub fn precompute_ratios(&mut self) {
        // path is none meaning that cands change every time
        if self.cfg.is_global_cand {
            let cands: Option<Candidates> = self.cfg.cand_path.as_ref().and_then(|path| {
                let content = std::fs::read(path).expect("invalid candidate file");
                serde_pickle::from_slice(&content, Default::default())
                    .ok()
                    .and_then(|cands| {
                        Some(Candidates {
                            num_cd: 5,
                            cands: crate::config::Cand::Flow(cands),
                        })
                    })
            });
            self.global_ratios = Some(self.pool.install(|| self.compute_ratios(&cands)));
        }
    }

    // pub fn precompute_ratios(&self, cands: &Option<Candidates>) -> Vec<Vec<(usize, f64, usize)>> {
    pub fn compute_ratios(&self, cands: &Option<Candidates>) -> Vec<Vec<(usize, f64, usize)>> {
        let mut result = Vec::new();

        // compute all the flows and their clusters first
        let num_nodes = self.graph.node_count();
        let num_cd = cands.as_ref().map(|c| c.num_cd).unwrap_or(num_nodes);
        let mut flows = Vec::new();
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                flows.push((i, j));
            }
        }

        // compute ratios per edge per flow [flow x edge x num_cd]
        let edges = &self.edges;
        for (f, &(src, dst)) in flows.iter().enumerate() {
            let mut ratios_per_flow = Vec::new();
            ratios_per_flow.par_extend((0..edges.len()).into_par_iter().flat_map_iter(|e| {
                // compute ratios of [flow f on edge e per cand]
                (0..num_cd)
                    .filter_map(|cd| {
                        let f_idx = self.nc.as_ref().map(|nc| nc[f]).unwrap_or(f);
                        // let f_idx = ncf[f];
                        let cand = get_candidate(cands, f_idx, src, cd);
                        let ratio = if cand == num_nodes + 1 || cand == num_nodes {
                            // shortest path
                            self.f(src, dst, edges[e])
                        } else {
                            // intermediate nodes
                            self.g(src, dst, cand, edges[e])
                        };
                        if ratio.abs() > f64::EPSILON {
                            //     v.push((demands[f] * ratio, action[&f_idx][cd]));
                            Some((e, ratio, cd))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            }));
            result.push(ratios_per_flow);
        }
        result
    }

    #[allow(dead_code)]
    pub fn compute_link_frac_with_action(&self, action: Vec<Vec<Vec<f64>>>) -> Vec<f64> {
        let node_count = self.graph.node_count();

        let mut link_loads = HashMap::new();
        for i in 0..node_count {
            for j in 0..node_count {
                for k in 0..node_count {
                    let ans = action[i][j][k];
                    if ans.abs() < f64::EPSILON {
                        continue;
                    }
                    for (e, cap) in self.edges.iter().zip(self.caps.iter()) {
                        let ratio = self.g(i, j, k, *e);
                        *link_loads.entry(*e).or_insert(0.0) += ratio * ans / *cap;
                    }
                }
            }
        }
        let loads = link_loads.into_values().collect::<Vec<f64>>();
        // println!("c{:?}", loads);
        println!(
            "computed MLU: {:?}",
            loads.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
        );
        loads
    }

    #[inline]
    pub fn f(&self, i: usize, j: usize, e: (usize, usize)) -> f64 {
        if !self.f_data.contains_key(&(i, j)) {
            0.0
        } else {
            let fracs = self.f_data.get(&(i, j)).unwrap();
            *fracs.get(&e).unwrap_or(&0.0)
        }
    }
    #[inline]
    pub fn g(&self, i: usize, j: usize, k: usize, e: (usize, usize)) -> f64 {
        self.f(i, k, e) + self.f(k, j, e)
    }

    pub fn solve_model(
        &self,
        tm: &TM,
        ratios: &Vec<Vec<(usize, f64, usize)>>,
        cands: Option<Candidates>, // candidate midpoints
        model_type: ModelType,
    ) -> Result<Solution, grb::Error> {
        // prepare data
        let num_nodes = tm.len();
        let tl = self.cfg.time_limit.unwrap_or(TLMAP[&self.cfg.toponame]);
        let num_thread = self.cfg.num_thread.unwrap_or_default();
        // flows, demands, [nc if necessary]
        let mut flows = Vec::new();
        let mut demands = Vec::new();
        let mut ncf = Vec::new();
        let mut edge_ratios = HashMap::new();

        for i in 0..num_nodes {
            if self.removed_nodes.contains(&i) {
                continue;
            }
            for j in 0..num_nodes {
                if i == j {
                    continue;
                }
                if tm[i][j].abs() > f64::EPSILON {
                    flows.push((i, j));
                    demands.push(tm[i][j].round());
                    // if need to apply node clustering
                    // handle nc based on flow -> ncf
                    let flow_idx = i * num_nodes + j;
                    // if no self.nc, then ncf is assert_eq none passed to lp function

                    // e, ratio, cd
                    let ratios_per_flow = &ratios[flow_idx];
                    let curr_idx = flows.len() - 1;

                    let c = self.nc.as_ref().map_or(curr_idx, |nc| nc[flow_idx]);
                    ncf.push(c);

                    for &(e, ratio, cd) in ratios_per_flow.iter() {
                        let v = edge_ratios.entry(e).or_insert(vec![]);
                        v.push((curr_idx, ratio, c, cd));
                    }
                }
            }
        }
        // caps, edges
        let edges = &self.edges;
        let caps = &self.caps;
        // cd (candidates), nc (node clustering)
        let ncf = self.cfg.num_nc.map(|_| ncf);

        let need_sol = self.cfg.need_solution;

        let removed_nodes = &self.removed_nodes;
        // solve
        match model_type {
            ModelType::ILP => ilp::solve_ilp(
                tl,
                num_thread,
                num_nodes,
                caps,
                edges,
                edge_ratios,
                flows,
                demands,
                cands,
                ncf,
                need_sol,
                removed_nodes
            ),
            ModelType::LP => lp::solve_lp(
                tl,
                num_thread,
                num_nodes,
                caps,
                edges,
                edge_ratios,
                flows,
                demands,
                cands,
                ncf,
                need_sol,
            ),
        }
    }
}
