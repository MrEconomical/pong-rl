use crate::config::{HEIGHT, WIDTH};
use std::sync::{Arc, Mutex};

use pixels::Pixels;

// Frame coordinate position struct

pub struct Point(usize, usize);

// Coordinate position change tracker struct

pub struct PosChange {
    prev: Point,
    current: Point,
}

// Pong game display frame buffer struct

pub struct Frame {
    prev: [[u8; WIDTH]; HEIGHT],
    current: [[u8; WIDTH]; HEIGHT],
    pixels: Option<Arc<Mutex<Pixels>>>,
}

impl Frame {
    // Initialize zeroed frame buffers with optional Pixels display

    pub fn new(pixels: Option<Arc<Mutex<Pixels>>>) -> Self {
        Self {
            prev: [[0; WIDTH]; HEIGHT],
            current: [[0; WIDTH]; HEIGHT],
            pixels,
        }
    }

    // Draw game state directly on current buffer assuming all buffers are zeroed

    pub fn force_draw(&mut self, ball: Point, left_paddle: Point, right_paddle: Point) {}

    // Clear buffers and reset Pixels display

    pub fn reset(&mut self) {
        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                self.prev[row][col] = 0;
                self.current[row][col] = 0;
            }
        }

        if let Some(pixels) = &self.pixels {
            let mut pixels = pixels.lock().unwrap();
            pixels.frame_mut().fill(0);
        }
    }
}
