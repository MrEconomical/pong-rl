use super::render;
use crate::config::{COLOR, PADDLE_HEIGHT, PADDLE_WIDTH};
use std::sync::{Arc, Mutex};

use pixels::Pixels;

// Frame coordinate position

#[derive(Clone, Copy, Default)]
pub struct Point(pub usize, pub usize);

#[derive(Clone, Copy, Default)]
pub struct FloatPoint(pub f64, pub f64);

// Game object position

#[derive(Clone, Copy, Default)]
struct ObjectState {
    ball: FloatPoint,
    ball_pos: Point,
    left_paddle: Point,
    right_paddle: Point,
}

impl ObjectState {
    // Batch set position values and calculate rounded ball position

    fn set_state(&mut self, ball: FloatPoint, left_paddle: Point, right_paddle: Point) {
        self.ball = ball;
        self.ball_pos = Point(ball.0.floor() as usize, ball.1.floor() as usize);
        self.left_paddle = left_paddle;
        self.right_paddle = right_paddle;
    }
}

// Pong game display frame

pub struct Frame {
    state: ObjectState,
    pixels: Option<Arc<Mutex<Pixels>>>,
}

impl Frame {
    // Create empty frame with optional Pixels display

    pub fn uninit(pixels: Option<Arc<Mutex<Pixels>>>) -> Self {
        if let Some(pixels) = &pixels {
            render::draw_border(pixels.lock().unwrap().frame_mut());
        }
        Self {
            state: ObjectState::default(),
            pixels,
        }
    }

    // Initialize frame state assuming zeroed buffer and render frame

    pub fn init_state(&mut self, ball: FloatPoint, left_paddle: Point, right_paddle: Point) {
        self.state.set_state(ball, left_paddle, right_paddle);
        if let Some(pixels) = &self.pixels {
            let mut pixels = pixels.lock().unwrap();
            Self::draw_state(pixels.frame_mut(), self.state, COLOR);
            pixels.render().expect("Error rendering frame");
        }
    }

    // Update game state with object positions and rerender

    pub fn update(&mut self, ball: FloatPoint, left_paddle: Point, right_paddle: Point) {
        if let Some(pixels) = &self.pixels {
            let mut pixels = pixels.lock().unwrap();
            Self::draw_state(pixels.frame_mut(), self.state, 0x00);
            self.state.set_state(ball, left_paddle, right_paddle);
            Self::draw_state(pixels.frame_mut(), self.state, COLOR);
            pixels.render().expect("Error rendering frame");
        } else {
            self.state.set_state(ball, left_paddle, right_paddle);
        }
    }

    // Reset game state and clear buffer without rerender

    pub fn reset(&mut self) {
        if let Some(pixels) = &self.pixels {
            let mut pixels = pixels.lock().unwrap();
            Self::draw_state(pixels.frame_mut(), self.state, 0x00);
        }
        self.state = ObjectState::default();
    }

    // Batch draw object state on Pixels RGBA frame

    fn draw_state(rgba_frame: &mut [u8], state: ObjectState, color: u8) {
        render::draw_ball(rgba_frame, state.ball, color);
        render::draw_rect(
            rgba_frame,
            state.left_paddle,
            PADDLE_WIDTH,
            PADDLE_HEIGHT,
            color,
        );
        render::draw_rect(
            rgba_frame,
            state.right_paddle,
            PADDLE_WIDTH,
            PADDLE_HEIGHT,
            color,
        );
    }
}
