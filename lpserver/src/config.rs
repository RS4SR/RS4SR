use clap::Parser;
use phf::phf_map;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// global config of SREnv
#[derive(Parser, Debug)]
#[clap(author, version, about)]
pub struct Config {
    #[clap(short, long, default_value = "1")]
    pub id: usize,
    #[clap(short, long, default_value = "GEANT")]
    pub toponame: String,
    #[clap(short, long, default_value = "0")]
    pub num_linkfail: usize,
    #[clap(short, long, default_value = "0")]
    pub num_linkadd: usize,
    #[clap(short, long, default_value = "0")]
    pub num_nodermv: usize,
    #[clap(short, long, default_value = "/home/wlh/coding/RS4SR2024/data/")]
    pub data_dir: PathBuf,
    #[clap(short, long, default_value = "/home/wlh/coding/RS4SR2024/nc/")]
    pub nc_dir: PathBuf,
    #[clap(long)]
    pub nc_method: Option<String>,
    #[clap(long, default_value = "24")]
    pub num_agents: usize,
    #[clap(long)]
    pub num_nc: Option<usize>,
    #[clap(long, default_value = "false")]
    pub is_global_cand: bool,
    #[clap(long)]
    pub cand_path: Option<String>,
    #[clap(long, default_value = "false")]
    pub need_solution: bool,
    #[clap(long)]
    pub time_limit: Option<f64>,
    #[clap(long, default_value= "1")]
    pub num_thread: Option<usize>,
}
impl Config {
    pub fn topo_path(&self) -> PathBuf {
        let mut path = self.data_dir.clone();
        path.push(&self.toponame);
        path
    }
    pub fn tm_path(&self) -> PathBuf {
        let mut path = self.data_dir.clone();
        path.push(format!("{}TM.pkl", &self.toponame));
        path
    }
    pub fn nc_path(&self) -> PathBuf {
        let mut path = self.nc_dir.clone();
        let nc_method = self.nc_method.as_ref().unwrap();
        path.push(format!(
            "{}-{nc_method}-nc-{}.pkl",
            &self.toponame,
            self.num_nc.unwrap()
        ));
        path
    }

    pub fn nc_path_none(&self) -> PathBuf {
        assert!(self.num_nc.is_none());
        let mut path = self.nc_dir.clone();
        let nc_method = self.nc_method.as_ref().unwrap();
        path.push(format!("{}-{nc_method}-nc-None.pkl", &self.toponame,));
        path
    }
}

pub static TLMAP: phf::Map<&'static str, f64> = phf_map! {
    "Abilene" => 1.0,
    "nobel" => 1.0,
    "GEANT" => 5.0,
    "germany50" => 50.0,
    "rf1755" => 500.0,
    "rf3967" => 500.0,
    "rf1221" => 1000.0,
    "rf1755g" => 500.0,
    "rf1755one" => 500.0,
    "rf6461" => 3000.0,
    "rf6461g" => 3000.0,
    "rf6461one" => 3000.0,
    "rf3257" => 10000.0,
    "rf1239" => 100000.0,
};

/// following are structs used to communicate with python code

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Cand {
    Network(Vec<usize>),   // the whole network share the same candidates
    Node(Vec<Vec<usize>>), // each node share the same candidates
    Flow(Vec<Vec<usize>>), // each real or fake flow has its own candidates
}

#[inline]
pub fn get_candidate(cands: &Option<Candidates>, f: usize, src: usize, cd: usize) -> usize {
    let cands = cands.as_ref().map(|c| &c.cands);
    match cands {
        None => cd,
        Some(Cand::Network(v)) => v[cd],
        Some(Cand::Node(v)) => v[src][cd],
        Some(Cand::Flow(v)) => v[f][cd],
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidates {
    pub num_cd: usize,
    pub cands: Cand,
}

#[derive(Serialize, Deserialize)]
pub struct Task {
    pub idx: Option<usize>,
    #[serde(flatten)]
    pub cands: Option<Candidates>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TEResult {
    pub idx: usize,
    pub mlu: f64,
    pub time: f64,
    pub action: Option<Vec<Vec<isize>>>,
}

pub struct Solution {
    pub mlu: f64,
    pub action: Option<Vec<Vec<isize>>>,
}

impl std::fmt::Debug for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cand_type = self.cands.as_ref().map(|c| match c.cands {
            Cand::Network(_) => "Network",
            Cand::Node(_) => "Node",
            Cand::Flow(_) => "Flow",
        });
        let num_cd = self.cands.as_ref().map(|c| c.num_cd);
        write!(
            f,
            "Task [ idx: {:?} cands: {:?} num_cd: {:?} ]",
            self.idx, cand_type, num_cd
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    LP,
    ILP,
}
