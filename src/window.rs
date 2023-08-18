use crate::config::{HEIGHT, WIDTH};
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
    Stop,
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
    event: Event<()>,
    control_flow: &mut ControlFlow,
    event_sender: &Sender<UserEvent>,
    pixels: &Mutex<Pixels>,
) {
    // Render on redraw requested

    if let Event::RedrawRequested(_) = event {
        if let Err(error) = pixels.lock().unwrap().render() {
            eprintln!("Error in render: {error}");
            let _ = event_sender.send(UserEvent::Exit);
            *control_flow = ControlFlow::Exit;
            return;
        }
    }

    // Check user input events

    if let Event::WindowEvent {
        event: window_event,
        ..
    } = event
    {
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

                if action_type == ElementState::Released {
                    event_sender
                        .send(UserEvent::PaddleInput(PaddleDir::Stop))
                        .unwrap();
                    return;
                } else {
                    match key {
                        VirtualKeyCode::W | VirtualKeyCode::Up => {
                            event_sender
                                .send(UserEvent::PaddleInput(PaddleDir::Up))
                                .unwrap();
                            return;
                        }
                        VirtualKeyCode::S | VirtualKeyCode::Down => {
                            event_sender
                                .send(UserEvent::PaddleInput(PaddleDir::Down))
                                .unwrap();
                            return;
                        }
                        _ => (),
                    }
                }
            }
            WindowEvent::Resized(size) => {
                // Resize Pixels surface on window resize

                if let Err(error) = pixels
                    .lock()
                    .unwrap()
                    .resize_surface(size.width, size.height)
                {
                    eprintln!("Error in resize: {error}");
                    let _ = event_sender.send(UserEvent::Exit);
                    *control_flow = ControlFlow::Exit;
                    return;
                }
            }
            WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                // Exit event loop on window close

                event_sender.send(UserEvent::Exit).unwrap();
                *control_flow = ControlFlow::Exit;
                return;
            }
            _ => (),
        }
    }
}
