use crate::config::{HEIGHT, WIDTH};
use std::sync::{Arc, Mutex};

use pixels::Pixels;

// Pong game display frame buffer struct

pub struct Frame {
    prev: [[u8; WIDTH]; HEIGHT],
    current: [[u8; WIDTH]; HEIGHT],
    pixels: Option<Arc<Mutex<Pixels>>>,
}

impl Frame {
    // Initialize frame buffers with optional Pixels display

    pub fn new(pixels: Option<Arc<Mutex<Pixels>>>) -> Self {
        Self {
            prev: [[0; WIDTH]; HEIGHT],
            current: [[0; WIDTH]; HEIGHT],
            pixels,
        }
    }
}
