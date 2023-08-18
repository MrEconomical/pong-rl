use crate::config::{HEIGHT, WIDTH};
use std::mem;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;

use pixels::{Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::Event;
use winit::event_loop::{ControlFlow, EventLoopBuilder};
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::WindowBuilder;

// Window event enum

#[derive(Clone, Copy, PartialEq)]
pub enum UserEvent {
    PaddleInput(PaddleDir),
    Exit,
}

// Paddle direction enum

#[derive(Clone, Copy, PartialEq)]
pub enum PaddleDir {
    Up,
    Down,
}

// Spawn window thread and return Pixels, window events channel, and events thread handle

pub fn create_window() -> (Arc<Mutex<Pixels>>, Receiver<UserEvent>, JoinHandle<()>) {
    // Create channels for receiving events and Pixels

    let (event_sender, event_receiver) = mpsc::channel();
    let (pixels_sender, pixels_receiver) = mpsc::channel();

    // Create thread and move sender channels into event handler task

    let handle = thread::spawn(move || build_window(event_sender, pixels_sender));
    let pixels = pixels_receiver.recv().expect("failed to receive Pixels");

    (pixels, event_receiver, handle)
}

// Build window and run event loop with event handler

fn build_window(event_sender: Sender<UserEvent>, pixels_sender: Sender<Arc<Mutex<Pixels>>>) {
    // Create event loop and window

    let event_loop = EventLoopBuilder::new().with_any_thread(true).build(); // Only works on Windows
    let window = {
        let default_size = LogicalSize::new((WIDTH * 2) as f64, (HEIGHT * 2) as f64);
        let min_size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Pong")
            .with_inner_size(default_size)
            .with_min_inner_size(min_size)
            .build(&event_loop)
            .unwrap()
    };

    // Create Pixels pixel buffer

    let pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        let pixels = Arc::new(Mutex::new(
            Pixels::new(WIDTH, HEIGHT, surface_texture).unwrap(),
        ));

        // Send Pixels to main thread

        pixels_sender
            .send(pixels.clone())
            .expect("failed to send Pixels");
        mem::drop(pixels_sender);

        pixels
    };

    // Run blocking event loop

    event_loop.run(move |event, _, control_flow| {
        handle_events(event, control_flow, &event_sender, &pixels);
    });
}

// Handle window events and send user input events to channel queue

fn handle_events(
    event: Event<()>,
    control_flow: &mut ControlFlow,
    event_sender: &Sender<UserEvent>,
    pixels: &Mutex<Pixels>,
) {
    // Render on redraw requested

    if let Event::RedrawRequested(_) = event {
        if let Err(error) = pixels.lock().unwrap().render() {
            eprintln!("Error in render: {error}");
            *control_flow = ControlFlow::Exit;
            return;
        }
    }
}
