use crate::core::Pong;
use crate::window;
use crate::window::{PaddleInput, UserEvent};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use pixels::Pixels;

// User-controlled Pong game struct

pub struct PongGame {
    pong: Pong,
    event_channel: Receiver<UserEvent>,
    pub window_handle: JoinHandle<()>,
}

impl PongGame {
    // Create window with event loop and initialize pong game

    pub fn new() -> Self {
        let (pixels, event_channel, window_handle) = window::create_window();
        Self {
            pong: Pong::new(Some(pixels)),
            event_channel,
            window_handle,
        }
    }
}
