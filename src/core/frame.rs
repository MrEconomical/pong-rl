use crate::config::{BALL_SIZE, COLOR, HEIGHT, WIDTH};
use std::sync::{Arc, Mutex};

use pixels::Pixels;

// Frame coordinate position struct

#[derive(Clone, Copy)]
pub struct Point(pub usize, pub usize);

// Coordinate position change tracker struct

#[derive(Clone, Copy)]
pub struct PosChange {
    prev: Point,
    current: Point,
}

// Pong game display frame buffer struct

pub struct Frame {
    prev: [u8; WIDTH * HEIGHT],
    current: [u8; WIDTH * HEIGHT],
    pixels: Option<Arc<Mutex<Pixels>>>,
}

impl Frame {
    // Initialize zeroed frame buffers with optional Pixels display

    pub fn new(pixels: Option<Arc<Mutex<Pixels>>>) -> Self {
        Self {
            prev: [0; WIDTH * HEIGHT],
            current: [0; WIDTH * HEIGHT],
            pixels,
        }
    }

    // Draw game state directly on current buffer assuming all buffers are zeroed

    pub fn force_draw(&mut self, ball: Point, left_paddle: Point, right_paddle: Point) {}

    // Clear buffers and reset Pixels display

    pub fn reset(&mut self, with_render: bool) {
        self.prev.fill(0);
        self.current.fill(0);

        if let Some(pixels) = &self.pixels {
            let mut pixels = pixels.lock().unwrap();
            pixels.frame_mut().fill(0);
            if with_render {
                pixels.render().expect("error rendering frame");
            }
        }
    }

    // Draw ball at position on buffer

    fn draw_ball(frame: &mut [u8], ball: Point) {
        for row in ball.1..ball.1 + BALL_SIZE {
            for col in ball.0..ball.0 + BALL_SIZE {
                // todo: figure out how the texture format works
            }
        }
    }
}
