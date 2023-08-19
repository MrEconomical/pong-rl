use pong_rl::FRAME_DELAY;
use pong_rl::{PongGame, TickResult};
use std::thread;
use std::time::Duration;

// Run user-controlled Pong game

fn main() {
    // Run games infinitely until user exit

    let mut pong = PongGame::new();

    'reset: loop {
        // Initialize game and pause

        pong.start();
        thread::sleep(Duration::from_millis(2000));

        loop {
            // Run game tick

            let tick_result = pong.tick();
            match tick_result {
                Some(TickResult::GameEnd) => {
                    thread::sleep(Duration::from_millis(2000));
                    break;
                }
                Some(TickResult::Exit) => break 'reset,
                _ => {
                    thread::sleep(Duration::from_millis(FRAME_DELAY));
                }
            }
        }
    }

    // Run window thread to completion

    pong.run_window();
}
