use super::render;
use crate::config::{BALL_SIZE, COLOR, HEIGHT, PADDLE_HEIGHT, PADDLE_WIDTH, WIDTH};
use std::sync::{Arc, Mutex};

use pixels::Pixels;

// Frame coordinate position structs

#[derive(Clone, Copy, Default)]
pub struct Point(pub usize, pub usize);
#[derive(Clone, Copy, Default)]
pub struct FloatPoint(pub f64, pub f64);

// Game object position struct

#[derive(Clone, Copy, Default)]
struct ObjectState {
    ball: FloatPoint,
    left_paddle: Point,
    right_paddle: Point,
}

impl ObjectState {
    // Batch set position values

    fn set_state(&mut self, ball: FloatPoint, left_paddle: Point, right_paddle: Point) {
        self.ball = ball;
        self.left_paddle = left_paddle;
        self.right_paddle = right_paddle;
    }
}

// Pong game display frame buffer struct

pub struct Frame {
    prev: Vec<u8>,
    prev_state: ObjectState,
    current: Vec<u8>,
    current_state: ObjectState,
    pixels: Option<Arc<Mutex<Pixels>>>,
}

impl Frame {
    // Initialize zeroed frame buffers with optional Pixels display

    pub fn zeroed(pixels: Option<Arc<Mutex<Pixels>>>) -> Self {
        Self {
            prev: vec![0; HEIGHT * WIDTH],
            prev_state: ObjectState::default(),
            current: vec![0; HEIGHT * WIDTH],
            current_state: ObjectState::default(),
            pixels,
        }
    }

    // Initialize frame state assuming zeroed buffers and render frame

    pub fn init_state(&mut self, ball: FloatPoint, left_paddle: Point, right_paddle: Point) {
        // Set internal frame states

        self.prev_state.set_state(ball, left_paddle, right_paddle);
        Self::draw_internal_state(&mut self.prev, self.prev_state, COLOR);
        self.current_state
            .set_state(ball, left_paddle, right_paddle);
        Self::draw_internal_state(&mut self.current, self.current_state, COLOR);

        // Set display frame state and render frame

        if let Some(pixels) = &self.pixels {
            let mut pixels = pixels.lock().unwrap();
            Self::draw_display_state(pixels.frame_mut(), self.current_state, COLOR);
            pixels.render().expect("Error rendering frame");
        }
    }

    // Update game state with object positions and rerender

    pub fn update(&mut self, ball: FloatPoint, left_paddle: Point, right_paddle: Point) {
        // Apply previous changes to previous state

        Self::draw_internal_state(&mut self.prev, self.prev_state, 0x00);
        Self::draw_internal_state(&mut self.prev, self.current_state, COLOR);
        self.prev_state = self.current_state;

        // Apply changes to current state

        self.current_state
            .set_state(ball, left_paddle, right_paddle);
        Self::draw_internal_state(&mut self.current, self.prev_state, 0x00);
        Self::draw_internal_state(&mut self.current, self.current_state, COLOR);

        // Render updated state

        if let Some(pixels) = &self.pixels {
            let mut pixels = pixels.lock().unwrap();
            Self::draw_display_state(pixels.frame_mut(), self.prev_state, 0x00);
            Self::draw_display_state(pixels.frame_mut(), self.current_state, COLOR);
            pixels.render().expect("Error rendering frame");
        }
    }

    // Reset game state and clear buffers without rerender

    pub fn reset(&mut self) {
        // Clear internal frames

        Self::draw_internal_state(&mut self.prev, self.prev_state, 0x00);
        self.prev_state = ObjectState::default();
        Self::draw_internal_state(&mut self.current, self.current_state, 0x00);

        // Clear display frame

        if let Some(pixels) = &self.pixels {
            let mut pixels = pixels.lock().unwrap();
            Self::draw_display_state(pixels.frame_mut(), self.current_state, COLOR);
        }

        self.current_state = ObjectState::default();
    }

    // Batch draw object state on internal frame

    fn draw_internal_state(frame: &mut [u8], state: ObjectState, color: u8) {
        render::draw_ball_internal(frame, state.ball, color);
        render::draw_internal(frame, state.left_paddle, PADDLE_WIDTH, PADDLE_HEIGHT, color);
        render::draw_internal(
            frame,
            state.right_paddle,
            PADDLE_WIDTH,
            PADDLE_HEIGHT,
            color,
        );
    }

    // Batch draw object state on Pixels RGBA frame

    fn draw_display_state(rgba_frame: &mut [u8], state: ObjectState, color: u8) {
        render::draw_ball_display(rgba_frame, state.ball, color);
        render::draw_display(
            rgba_frame,
            state.left_paddle,
            PADDLE_WIDTH,
            PADDLE_HEIGHT,
            color,
        );
        render::draw_display(
            rgba_frame,
            state.right_paddle,
            PADDLE_WIDTH,
            PADDLE_HEIGHT,
            color,
        );
    }
}
