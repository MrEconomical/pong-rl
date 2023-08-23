use crate::core::Pong;
use crate::window;
use crate::window::UserEvent;
use std::sync::mpsc::Receiver;
use std::thread::JoinHandle;

use pyo3::{pyclass, pymethods};

// Python-controlled Pong environment

#[pyclass]
pub struct PongEnv {
    pong: Pong,
    render: bool,
}

// Methods exposed to Python

#[pymethods]
impl PongEnv {
    // Create new Pong environment with rendering

    #[staticmethod]
    fn with_render() -> Self {
        let (pixels, _, _) = window::create_window();
        Self {
            pong: Pong::new(Some(pixels)),
            render: true,
        }
    }

    // Create new Pong environment without rendering

    #[staticmethod]
    fn without_render() -> Self {
        Self {
            pong: Pong::new(None),
            render: false,
        }
    }
}
