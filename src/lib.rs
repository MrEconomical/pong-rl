mod config;
mod core;
mod env;
mod game;
mod window;

pub use config::FRAME_DELAY;
pub use game::{PongGame, TickResult};

use env::PongEnv;

use pyo3::prelude::*;

// Export Pong game environment to Python

#[pymodule]
fn pong_rl(_: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<PongEnv>()?;
    Ok(())
}
