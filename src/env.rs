use crate::core::Pong;

use pyo3::{pyclass, pymethods};

// Python-controlled Pong environment struct

#[pyclass]
pub struct PongEnv {}

// Methods exposed to Python

#[pymethods]
impl PongEnv {
    #[new]
    fn new() -> Self {
        Self {}
    }

    fn do_something(&self) {
        for _ in 0..5 {
            println!("did something");
        }
    }
}
