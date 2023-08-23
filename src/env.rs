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
    event_channel: Option<Receiver<UserEvent>>,
    window_handle: Option<JoinHandle<()>>,
}

// Methods exposed to Python

#[pymethods]
impl PongEnv {
    // Create new pong environment with rendering

    #[staticmethod]
    fn with_render() -> Self {
        let (pixels, event_channel, window_handle) = window::create_window();
        Self {
            pong: Pong::new(Some(pixels)),
            render: true,
            event_channel: Some(event_channel),
            window_handle: Some(window_handle),
        }
    }

    // Create new pong environment without rendering

    #[staticmethod]
    fn without_render() -> Self {
        Self {
            pong: Pong::new(None),
            render: false,
            event_channel: None,
            window_handle: None,
        }
    }

    // Run window thread to completion

    pub fn run_window(&mut self) {
        assert!(self.render, "Instance does not have rendering enabled");
        self.window_handle
            .take()
            .expect("Error unwrapping window thread")
            .join()
            .expect("Error running window thread to completion");
    }
}
