use crate::config::{HEIGHT, WIDTH};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;

use pixels::{Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event_loop::{ControlFlow, EventLoopBuilder};
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::WindowBuilder;

// Window event enum

#[derive(Clone, Copy, PartialEq)]
pub enum WindowEvent {
    PaddleInput(PaddleDir),
    Exit,
}

// Paddle direction enum

#[derive(Clone, Copy, PartialEq)]
pub enum PaddleDir {
    Up,
    Down,
}

// Spawn window thread and return Pixels pixel buffer, window events channel, and events thread handle

pub fn create_window() -> (Arc<Mutex<Pixels>>, Receiver<WindowEvent>, JoinHandle<()>) {
    // Create channels for receiving events and Pixels struct

    let (event_sender, event_receiver) = mpsc::channel();
    let (pixels_sender, pixels_receiver) = mpsc::channel();

    // Create thread and move sender channels into event handler task

    let handle = thread::spawn(move || handle_events(event_sender, pixels_sender));
    let pixels = pixels_receiver.recv().expect("failed to receive Pixels");

    (pixels, event_receiver, handle)
}

// Send user input events through channel to queue game events

fn handle_events(event_sender: Sender<WindowEvent>, pixels_sender: Sender<Arc<Mutex<Pixels>>>) {
    // Create event loop and window

    let event_loop = EventLoopBuilder::new().with_any_thread(true).build(); // Only works on Windows
    let window = {
        let default_size = LogicalSize::new((WIDTH * 3) as f64, (HEIGHT * 3) as f64);
        let min_size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Pong")
            .with_inner_size(default_size)
            .with_min_inner_size(min_size)
            .build(&event_loop)
            .unwrap()
    };
}
