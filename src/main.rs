fn main() {
    let (_, _, handle) = pong_rl::create_window();
    let _ = handle.join();
}
