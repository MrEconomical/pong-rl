use crate::config::{BALL_SIZE, COLOR, HEIGHT, PADDLE_HEIGHT, PADDLE_WIDTH, WIDTH};
use std::sync::{Arc, Mutex};

use pixels::Pixels;

// Frame coordinate position struct

#[derive(Clone, Copy, Default, PartialEq)]
pub struct Point(pub usize, pub usize);

// Game object position struct

#[derive(Clone, Copy, Default)]
struct ObjectState {
    ball: Point,
    left_paddle: Point,
    right_paddle: Point,
}

impl ObjectState {
    // Batch set position values

    fn set_state(&mut self, ball: Point, left_paddle: Point, right_paddle: Point) {
        self.ball = ball;
        self.left_paddle = left_paddle;
        self.right_paddle = right_paddle;
    }
}

// Pong game display frame buffer struct

pub struct Frame {
    prev: [u8; HEIGHT * WIDTH],
    prev_state: ObjectState,
    current: [u8; HEIGHT * WIDTH],
    current_state: ObjectState,
    pixels: Option<Arc<Mutex<Pixels>>>,
}

impl Frame {
    // Initialize zeroed frame buffers with optional Pixels display

    pub fn zeroed(pixels: Option<Arc<Mutex<Pixels>>>) -> Self {
        Self {
            prev: [0; HEIGHT * WIDTH],
            prev_state: ObjectState::default(),
            current: [0; HEIGHT * WIDTH],
            current_state: ObjectState::default(),
            pixels,
        }
    }

    // Initialize frame state assuming zeroed buffers and render frame

    pub fn init_state(&mut self, ball: Point, left_paddle: Point, right_paddle: Point) {
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

    pub fn update(&mut self, ball: Point, left_paddle: Point, right_paddle: Point) {
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
        Self::draw_internal(frame, state.ball, BALL_SIZE, BALL_SIZE, color);
        Self::draw_internal(frame, state.left_paddle, PADDLE_WIDTH, PADDLE_HEIGHT, color);
        Self::draw_internal(
            frame,
            state.right_paddle,
            PADDLE_WIDTH,
            PADDLE_HEIGHT,
            color,
        );
    }

    // Batch draw object state on Pixels RGBA frame

    fn draw_display_state(rgba_frame: &mut [u8], state: ObjectState, color: u8) {
        Self::draw_display(rgba_frame, state.ball, BALL_SIZE, BALL_SIZE, color);
        Self::draw_display(
            rgba_frame,
            state.left_paddle,
            PADDLE_WIDTH,
            PADDLE_HEIGHT,
            color,
        );
        Self::draw_display(
            rgba_frame,
            state.right_paddle,
            PADDLE_WIDTH,
            PADDLE_HEIGHT,
            color,
        );
    }

    // Draw rectangle on internal frame at position with width and height

    fn draw_internal(frame: &mut [u8], pos: Point, width: usize, height: usize, color: u8) {
        for row in pos.1..pos.1 + height {
            for col in pos.0..pos.0 + width {
                frame[row * WIDTH + col] = color;
            }
        }
    }

    // Draw rectangle on Pixels RGBA frame at position with width and height

    fn draw_display(rgba_frame: &mut [u8], pos: Point, width: usize, height: usize, color: u8) {
        for row in pos.1..pos.1 + height {
            for col in pos.0..pos.0 + width {
                let offset = (row * WIDTH + col) * 4;
                rgba_frame[offset] = color;
                rgba_frame[offset + 1] = color;
                rgba_frame[offset + 2] = color;
                rgba_frame[offset + 3] = 0xFF;
            }
        }
    }
}
