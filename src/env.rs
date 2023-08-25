use crate::config::{BALL_SPEED, HEIGHT, WIDTH};
use crate::core::{GameResult, PaddleMove, Pong};
use crate::window;
use crate::window::UserEvent;
use std::sync::mpsc::Receiver;

use numpy::PyArray1;
use pyo3::{pyclass, pymethods, Python};

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

    // Normalize full internal game state

    fn get_normalized_state<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        let state = self.pong.get_game_state();
        PyArray1::from_slice(
            py,
            &[
                (state[0] / WIDTH as f64 - 0.5) * 2.0,  // Ball x position
                (state[1] / HEIGHT as f64 - 0.5) * 2.0, // Ball y position
                state[2] / BALL_SPEED as f64,           // Ball x velocity
                state[3] / BALL_SPEED as f64,           // Ball y velocity
                (state[4] / HEIGHT as f64 - 0.5) * 2.0, // Left paddle y position
                (state[5] / HEIGHT as f64 - 0.5) * 2.0, // Right paddle y position
            ],
        )
    }

    // Reset game to initial state

    pub fn reset(&mut self) {
        self.pong.clear_game();
    }
}
