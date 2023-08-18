mod config;
mod core;
mod env;
mod game;

pub use game::PongGame;

use env::PongEnv;
use pyo3::prelude::*;

#[pymodule]
fn pong_rl(_: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<PongEnv>()?;
    Ok(())
}
