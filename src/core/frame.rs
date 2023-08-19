use crate::config::{BALL_SIZE, COLOR, HEIGHT, WIDTH};
use std::sync::{Arc, Mutex};

use pixels::Pixels;

// Frame coordinate position struct

#[derive(Clone, Copy, Default, PartialEq)]
pub struct Point(pub usize, pub usize);

// Game object position struct

#[derive(Default)]
struct ObjectState {
    ball: Point,
    left_paddle: Point,
    right_paddle: Point,
}

// Pong game display frame buffer struct

pub struct Frame {
    prev: [u8; WIDTH * HEIGHT],
    prev_state: ObjectState,
    current: [u8; WIDTH * HEIGHT],
    current_state: ObjectState,
    pixels: Option<Arc<Mutex<Pixels>>>,
}

impl Frame {
    // Initialize zeroed frame buffers with optional Pixels display

    pub fn zeroed(pixels: Option<Arc<Mutex<Pixels>>>) -> Self {
        Self {
            prev: [0; WIDTH * HEIGHT],
            prev_state: ObjectState::default(),
            current: [0; WIDTH * HEIGHT],
            current_state: ObjectState::default(),
            pixels,
        }
    }
}
