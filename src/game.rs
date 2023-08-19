use crate::core::{GameResult, PaddleMove, Pong};
use crate::window;
use crate::window::{PaddleInput, UserEvent};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use pixels::Pixels;

// Game tick result enum

pub enum TickResult {
    GameEnd,
    Exit,
}

// User-controlled Pong game struct

pub struct PongGame {
    pong: Pong,
    event_channel: Receiver<UserEvent>,
    window_handle: JoinHandle<()>,
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

    // Clear input buffer and start pong game with initial state

    pub fn start(&mut self) {
        self.pong.start_game();
    }

    // Advance game and return game state

    pub fn tick(&mut self) -> Option<TickResult> {
        unimplemented!();
    }

    // Reset game to initial state

    pub fn reset(&mut self) {
        self.pong.clear_game();
    }

    // Run window thread to completion

    pub fn run_window(self) {
        self.window_handle
            .join()
            .expect("Error running window thread to completion");
    }
}
