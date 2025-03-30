// use std::time::Instant;
use grb::prelude::*;
use std::collections::HashMap;

use crate::config::{Candidates, Solution};

pub fn solve_lp(
    tl: f64,                  // time limit
    num_thread: usize,
    num_n: usize,             // node numbers
    caps: &[f64],             // capacity 1xe
    edges: &[(usize, usize)], // edges ex2
    edge_ratios: HashMap<usize, Vec<(usize, f64, usize, usize)>>,
    flows: Vec<(usize, usize)>, // flows fx2
    demands: Vec<f64>,          // demands 1xf
    cands: Option<Candidates>,  // candidate midpoints fxcd
    nc: Option<Vec<usize>>,             // node clusters fx1
    need_sol: bool,             // if need to return full solution
) -> Result<Solution, grb::Error> {
    // 0. initilize input
    let num_f = flows.len();
    let num_cd = cands.as_ref().map(|c| c.num_cd).unwrap_or(num_n);

    // 1. create model
    let env = Env::new("")?; // mute gurobi.log
    let mut model = Model::with_env("mlu_lp", &env)?;
    model.set_param(param::Threads, num_thread as i32)?;
    model.set_param(param::Method, 2)?;
    model.set_param(param::TimeLimit, tl)?;
    model.set_param(param::OutputFlag, 0)?;

    // 2. create decision variables
    let theta: Var = add_ctsvar!(model, name: "theta", bounds: 0.0..)?;
    let mut action: HashMap<usize, Vec<Var>> = HashMap::new();
    for f in 0..num_f {
        let f_idx = nc.as_ref().map_or(f, |nc| nc[f]);
        // let f_idx = nc[f];
        if action.contains_key(&f_idx) {
            continue;
        }
        let mut vs = Vec::new();
        for cd in 0..num_cd {
            let name = &format!("action_{}_{}", f, cd);
            let var = add_ctsvar!(model, name: name)?;
            vs.push(var);
        }
        action.insert(f_idx, vs);
    }

    // 3. demand constraints
    for f in action.keys() {
        let con = c!(action[f].iter().grb_sum() >= 1);
        let name = &format!("con_d{}", f);
        model.add_constr(name, con)?;
    }

    // 4. utilization constraints
    for e in 0..edges.len() {
        if edge_ratios.contains_key(&e) {
            let con = c!(edge_ratios[&e]
                .iter()
                .map(|&(f, ratio, f_idx, cd)| { 
                    demands[f] * ratio * action[&f_idx][cd] 
                })
                .grb_sum()
                <= theta * caps[e]);
            let name = &format!("con_e{}", e);
            model.add_constr(name, con)?;
        }
    }

    // 5. set objective and optimze model
    model.set_objective(theta, grb::ModelSense::Minimize)?;
    model.optimize()?;

    // 6. get solution
    let mlu = model.get_attr(attr::ObjVal)?;
    let sol = if need_sol {
        let mut sol = vec![vec![-1; num_n]; num_n];
        for (f, (src, dst)) in flows.iter().enumerate() {
            let f_sol = model.get_obj_attr_batch(attr::X, action[&f].clone())?;
            for (cd, ans) in f_sol.iter().enumerate() {
                if ans.abs() > f64::EPSILON {
                    sol[*src][*dst] = cd as isize;
                }
            }
        }
        Some(sol)
    } else {
        None
    };

    let sol = Solution { mlu, action: sol };
    Ok(sol)
}
