mod config;
mod core;
mod env;
mod game;
mod window;

pub use game::PongGame;

use env::PongEnv;

use pyo3::prelude::*;

// Export Pong game environment to Python

#[pymodule]
fn pong_rl(_: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<PongEnv>()?;
    Ok(())
}
