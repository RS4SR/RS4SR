use std::time::Instant;

use crate::config::{Candidates, Config, ModelType, TEResult, Task};
use crate::te::{self, TESolver, TM};

const TYPE: ModelType = ModelType::ILP;

fn _pickle_solution(sol: TEResult) -> Result<Vec<u8>, serde_pickle::error::Error> {
    let res = serde_pickle::to_vec(&sol, Default::default())?;
    Ok(res)
}

fn _unpickle_task(buf: &[u8]) -> Result<Task, serde_pickle::error::Error> {
    let task: Task = serde_pickle::from_slice(buf, Default::default())?;
    Ok(task)
}

pub struct TEServer {
    id: usize,
    tms: Vec<TM>,
    solver: TESolver,
}

impl TEServer {
    pub fn new(cfg: Config) -> TEServer {
        let id = cfg.id;
        let tms = te::load_tm_list(cfg.tm_path());
        println!("Creating LPServer [{:?}]", &cfg);
        let solver = TESolver::new(cfg);
        TEServer { id, tms, solver }
    }

    pub fn solve(
        &self,
        idx: usize,
        ratios: Option<Vec<Vec<(usize, f64, usize)>>>,
        cands: Option<Candidates>,
    ) -> Result<TEResult, grb::Error> {
        let tm = &self.tms[idx];

        let ratios = if ratios.is_some() {
            ratios.as_ref().unwrap()
        } else {
            self.solver.global_ratios.as_ref().unwrap()
        };

        // solve and record time
        let now = Instant::now();
        // WARNING: watch out!
        let sol = self.solver.solve_model(tm, ratios, cands, TYPE)?;
        let (mlu, action) = (sol.mlu, sol.action);
        let time = now.elapsed().as_secs_f64() + self.solver.ratio_compute_time;

        let result = TEResult {
            idx,
            mlu,
            time,
            action,
        };
        println!("Solve TM idx: {idx} Ans: {mlu:.3} Time: {time:.3} s");
        Ok(result)
    }

    pub fn run(self, num_threads: usize) -> Result<(), Box<dyn std::error::Error>> {
        // create zmq socket
        let id = self.id;
        let ctx = zmq::Context::new();
        let pull_sock = ctx.socket(zmq::PULL)?;
        pull_sock.connect(&format!("ipc:///tmp/tasks{id}"))?;
        let push_sock = ctx.socket(zmq::PUSH)?;
        push_sock.connect(&format!("ipc:///tmp/results{id}"))?;

        // create thread pool
        let this = std::sync::Arc::new(self);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()?;
        let (tx, rx) = std::sync::mpsc::channel();
        println!("Prepare done. Start to listen...");

        // spawn thread to send result
        let handle = std::thread::spawn(move || {
            for solution in rx {
                let ans = _pickle_solution(solution).unwrap();
                push_sock.send(ans, 0).ok();
            }
            push_sock.send("quit", 0).ok();
        });

        // handle tasks
        loop {
            let msg = pull_sock.recv_bytes(0)?;
            let task = _unpickle_task(&msg)?;
            let tx = tx.clone();
            let this = this.clone();
            println!("Recv task: {:?}", task);
            if task.idx.is_none() {
                break;
            }

            let now = Instant::now();
            let ratios = match this.solver.global_ratios {
                Some(_) => None,
                None => Some(
                    this.solver
                        .pool
                        .install(|| this.solver.compute_ratios(&task.cands)),
                ),
            };
            let local_ratio_time = now.elapsed().as_secs_f64();

            pool.spawn(
                move || match this.solve(task.idx.unwrap(), ratios, task.cands) {
                    Ok(mut result) => {
                        result.time += local_ratio_time;
                        tx.send(result).unwrap()
                    }
                    Err(e) => eprintln!("Solve task {:?} failed, error: {}", task.idx, e),
                },
            );
        }
        drop(tx); // need to close all senders
        handle.join().ok();
        Ok(())
    }
}
