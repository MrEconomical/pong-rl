fn main() {
    let (_, event_channel, handle) = pong_rl::create_window();
    let _ = handle.join();
}
