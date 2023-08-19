use crate::config::{BALL_SIZE, COLOR, HEIGHT, PADDLE_HEIGHT, PADDLE_WIDTH, WIDTH};
use std::sync::{Arc, Mutex};

use pixels::Pixels;

// Frame coordinate position struct

#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Point(pub usize, pub usize);

// Game object position struct

#[derive(Default)]
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
        Self::draw_internal(&mut self.prev, ball, BALL_SIZE, BALL_SIZE);
        Self::draw_internal(&mut self.prev, left_paddle, PADDLE_WIDTH, PADDLE_HEIGHT);
        Self::draw_internal(&mut self.prev, right_paddle, PADDLE_WIDTH, PADDLE_HEIGHT);

        self.current_state
            .set_state(ball, left_paddle, right_paddle);
        Self::draw_internal(&mut self.current, ball, BALL_SIZE, BALL_SIZE);
        Self::draw_internal(&mut self.current, left_paddle, PADDLE_WIDTH, PADDLE_HEIGHT);
        Self::draw_internal(&mut self.current, right_paddle, PADDLE_WIDTH, PADDLE_HEIGHT);

        // Set display frame state and render frame

        if let Some(pixels) = &self.pixels {
            let mut pixels = pixels.lock().unwrap();
            let pixels_frame = pixels.frame_mut();
            Self::draw_display(pixels_frame, ball, BALL_SIZE, BALL_SIZE);
            Self::draw_display(pixels_frame, left_paddle, PADDLE_WIDTH, PADDLE_HEIGHT);
            Self::draw_display(pixels_frame, right_paddle, PADDLE_WIDTH, PADDLE_HEIGHT);
            pixels.render().expect("Error rendering frame");
        }
    }

    // Update game state with object positions and rerender

    pub fn update(&mut self, ball: Point, left_paddle: Point, right_paddle: Point) {}

    // Reset game state and clear buffers without rerender

    pub fn reset(&mut self) {}

    // Draw rectangle on internal frame at position with width and height

    fn draw_internal(frame: &mut [u8], pos: Point, width: usize, height: usize) {
        for row in pos.1..pos.1 + height {
            for col in pos.0..pos.0 + width {
                frame[row * WIDTH + col] = COLOR;
            }
        }
    }

    // Draw rectangle on Pixels RGBA frame at position with width and height

    fn draw_display(rgba_frame: &mut [u8], pos: Point, width: usize, height: usize) {
        for row in pos.1..pos.1 + height {
            for col in pos.0..pos.0 + width {
                let offset = (row * WIDTH + col) * 4;
                rgba_frame[offset] = COLOR;
                rgba_frame[offset + 1] = COLOR;
                rgba_frame[offset + 2] = COLOR;
                rgba_frame[offset + 3] = 0xFF;
            }
        }
    }
}
