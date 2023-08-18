use pong_rl::PongGame;

fn main() {
    let pong = PongGame::new();
    pong.window_handle.join();
}
