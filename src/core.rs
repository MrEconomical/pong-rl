mod frame;

use frame::{Frame, Point, PosChange};
use std::sync::{Arc, Mutex};

use pixels::Pixels;

// Core Pong game struct

pub struct Pong {
    frame: Frame,
}

impl Pong {
    // Create pong game with optional Pixels display

    pub fn new(pixels: Option<Arc<Mutex<Pixels>>>) -> Self {
        Pong {
            frame: Frame::new(pixels),
        }
    }
}
