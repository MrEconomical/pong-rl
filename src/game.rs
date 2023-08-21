use crate::core::{GameResult, PaddleMove, Pong};
use crate::window;
use crate::window::{PaddleInput, UserEvent};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use pixels::Pixels;

// Active paddle moves selected by key presses

struct ActiveMoves {
    up: bool,
    down: bool,
}

// Game tick result

pub enum TickResult {
    GameEnd,
    Exit,
}

// User-controlled Pong game

pub struct PongGame {
    pong: Pong,
    selected_move: Option<PaddleMove>,
    active_moves: ActiveMoves,
    event_channel: Receiver<UserEvent>,
    window_handle: JoinHandle<()>,
}

impl PongGame {
    // Create window with event loop and initialize pong game

    pub fn new() -> Self {
        let (pixels, event_channel, window_handle) = window::create_window();
        Self {
            pong: Pong::new(Some(pixels)),
            selected_move: None,
            active_moves: ActiveMoves {
                up: false,
                down: false,
            },
            event_channel,
            window_handle,
        }
    }

    // Clear input buffer and start pong game with initial state

    pub fn start(&mut self) -> bool {
        let exited = self.process_events();
        if exited {
            return true;
        }

        self.pong.start_game();
        false
    }

    // Process channel events and return if the user exited

    pub fn process_events(&mut self) -> bool {
        while let Ok(event) = self.event_channel.try_recv() {
            match event {
                UserEvent::GameInput(paddle_input) => match paddle_input {
                    PaddleInput::Up => {
                        self.active_moves.up = false;
                        if matches!(self.selected_move, Some(PaddleMove::Up)) {}
                    }
                    _ => (),
                },
                UserEvent::Exit => return true,
            }
        }
        return false;
    }

    // Advance game and return game state

    pub fn tick(&mut self) -> Option<TickResult> {
        let exited = self.process_events();
        if exited {
            return Some(TickResult::Exit);
        }

        let game_result = self.pong.tick(self.selected_move);
        if game_result.is_some() {
            Some(TickResult::GameEnd)
        } else {
            None
        }
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
