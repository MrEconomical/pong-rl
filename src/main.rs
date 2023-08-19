use pong_rl::PongGame;

fn main() {
    let mut pong = PongGame::new();
    pong.start();
    pong.window_handle.join();
}
