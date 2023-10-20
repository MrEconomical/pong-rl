use crate::config::{TOTAL_HEIGHT, TOTAL_WIDTH, WINDOW_SCALE};
use std::mem;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;

use pixels::{Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoopBuilder};
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::WindowBuilder;

// Window event

#[derive(Clone, Copy)]
pub enum UserEvent {
    GameInput(PaddleInput),
    Exit,
}

// Paddle input

#[derive(Clone, Copy)]
pub enum PaddleInput {
    Up,
    StopUp,
    Down,
    StopDown,
}

// Spawn window thread and return Pixels, window events channel, and events thread handle

pub fn create_window() -> (Arc<Mutex<Pixels>>, Receiver<UserEvent>, JoinHandle<()>) {
    // Create channels for receiving events and Pixels

    let (event_sender, event_receiver) = mpsc::channel();
    let (pixels_sender, pixels_receiver) = mpsc::channel();

    // Create thread and move sender channels into event handler task

    let handle = thread::spawn(move || build_window(event_sender, pixels_sender));
    let pixels = pixels_receiver.recv().unwrap();

    (pixels, event_receiver, handle)
}

// Build window and run event loop with event handler

fn build_window(event_sender: Sender<UserEvent>, pixels_sender: Sender<Arc<Mutex<Pixels>>>) {
    // Create event loop and window

    let event_loop = EventLoopBuilder::new().with_any_thread(true).build(); // Only works on Windows
    let window = {
        let default_size = LogicalSize::new(
            (TOTAL_WIDTH * WINDOW_SCALE) as f64,
            (TOTAL_HEIGHT * WINDOW_SCALE) as f64,
        );
        let min_size = LogicalSize::new(TOTAL_WIDTH as f64, TOTAL_HEIGHT as f64);
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
            Pixels::new(TOTAL_WIDTH as u32, TOTAL_HEIGHT as u32, surface_texture).unwrap(),
        ));

        // Send Pixels to main thread

        pixels_sender.send(pixels.clone()).unwrap();
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
    event: Event<'_, ()>,
    control_flow: &mut ControlFlow,
    event_sender: &Sender<UserEvent>,
    pixels: &Mutex<Pixels>,
) {
    // Render on redraw requested

    if matches!(event, Event::RedrawRequested(_)) {
        if let Err(error) = pixels.lock().unwrap().render() {
            eprintln!("Error rendering on redraw: {error}");
            let _ = event_sender.send(UserEvent::Exit);
            *control_flow = ControlFlow::Exit;
            return;
        }
    }

    // Check user input events

    let window_event = match event {
        Event::WindowEvent {
            event: window_event,
            ..
        } => window_event,
        _ => return,
    };

    match window_event {
        WindowEvent::KeyboardInput {
            input:
                KeyboardInput {
                    state: action_type,
                    virtual_keycode: Some(key),
                    ..
                },
            ..
        } => {
            // Send paddle direction input to channel

            match key {
                VirtualKeyCode::W | VirtualKeyCode::Up => {
                    event_sender
                        .send(if matches!(action_type, ElementState::Pressed) {
                            UserEvent::GameInput(PaddleInput::Up)
                        } else {
                            UserEvent::GameInput(PaddleInput::StopUp)
                        })
                        .unwrap();
                }
                VirtualKeyCode::S | VirtualKeyCode::Down => {
                    event_sender
                        .send(if matches!(action_type, ElementState::Pressed) {
                            UserEvent::GameInput(PaddleInput::Down)
                        } else {
                            UserEvent::GameInput(PaddleInput::StopDown)
                        })
                        .unwrap();
                }
                _ => (),
            }
        }
        WindowEvent::Resized(size) => {
            // Resize Pixels surface on window resize

            if let Err(error) = pixels
                .lock()
                .unwrap()
                .resize_surface(size.width, size.height)
            {
                eprintln!("Error rendering on resize: {error}");
                let _ = event_sender.send(UserEvent::Exit);
                *control_flow = ControlFlow::Exit;
            }
        }
        WindowEvent::CloseRequested | WindowEvent::Destroyed => {
            // Exit event loop on window close

            event_sender.send(UserEvent::Exit).unwrap();
            *control_flow = ControlFlow::Exit;
        }
        _ => (),
    }
}
