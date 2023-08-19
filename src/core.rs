mod frame;

use frame::{Frame, Point};
use std::sync::{Arc, Mutex};

use pixels::Pixels;

// Fractional position struct

#[derive(Clone, Copy)]
struct FloatPoint(pub f64, pub f64);

// Ball velocity struct

#[derive(Clone, Copy)]
struct Velocity {
    pub x: f64,
    pub y: f64,
}

// Core Pong game struct

pub struct Pong {
    ball: FloatPoint,
    ball_velocity: Velocity,
    left_paddle: Point,
    right_paddle: Point,
    frame: Frame,
}

impl Pong {
    // Create pong game with optional Pixels display

    pub fn new(pixels: Option<Arc<Mutex<Pixels>>>) -> Self {
        Pong {
            ball: Pong::initial_ball_pos(),
            ball_velocity: Pong::random_ball_velocity(),
            left_paddle: Pong::initial_left_paddle_pos(),
            right_paddle: Pong::initial_right_paddle_pos(),
            frame: Frame::new(pixels),
        }
    }

    // Reset game and frame buffers

    pub fn reset(&mut self) {
        // todo: reset frame
        self.ball = Pong::initial_ball_pos();
        self.ball_velocity = Pong::random_ball_velocity();
        self.left_paddle = Pong::initial_left_paddle_pos();
        self.right_paddle = Pong::initial_right_paddle_pos();
        // todo: draw on frame
    }

    // Calculate initial game values

    fn initial_ball_pos() -> FloatPoint {
        FloatPoint(0.0, 0.0)
    }

    fn random_ball_velocity() -> Velocity {
        Velocity { x: 0.0, y: 0.0 }
    }

    fn initial_left_paddle_pos() -> Point {
        Point(0, 0)
    }

    fn initial_right_paddle_pos() -> Point {
        Point(0, 0)
    }
}
