use crate::core::{GameResult, PaddleMove, Pong};
use crate::window;
use crate::window::UserEvent;
use std::sync::mpsc::Receiver;

use pyo3::{pyclass, pymethods};

// Python-controlled Pong environment

#[pyclass]
pub struct PongEnv {
    pong: Pong,
    _event_channel: Option<Receiver<UserEvent>>,
}

// Methods exposed to Python

#[pymethods]
impl PongEnv {
    // Create new Pong environment with rendering

    #[staticmethod]
    fn with_render() -> Self {
        let (pixels, event_channel, _) = window::create_window();
        Self {
            pong: Pong::new(Some(pixels)),
            _event_channel: Some(event_channel),
        }
    }

    // Create new Pong environment without rendering

    #[staticmethod]
    fn without_render() -> Self {
        Self {
            pong: Pong::new(None),
            _event_channel: None,
        }
    }

    // Start game with initial state

    fn start(&mut self) {
        self.pong.start_game();
    }

    // Advance game with action and return reward

    fn tick(&mut self, action: i32) -> i32 {
        let input = if action == 0 {
            Some(PaddleMove::Down)
        } else {
            Some(PaddleMove::Up)
        };

        let game_result = self.pong.tick(input);
        match game_result {
            Some(GameResult::Win) => 1,
            Some(GameResult::Lose) => -1,
            None => 0,
        }
    }

    // Reset game to initial state

    pub fn reset(&mut self) {
        self.pong.clear_game();
    }
}
