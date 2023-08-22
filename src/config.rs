// Game configuration parameters

pub const WIDTH: usize = 800;
pub const HEIGHT: usize = 480;
pub const RESCALE: usize = 4;
pub const FRAME_DELAY: u64 = 33;

pub const BALL_SIZE: usize = 14;
pub const BALL_SPEED: f64 = 14.0;
pub const MAX_BOUNCE_ANGLE: f64 = 65.0;
pub const MAX_INITIAL_ANGLE: f64 = 45.0;

pub const PADDLE_WIDTH: usize = 10;
pub const PADDLE_HEIGHT: usize = 80;
pub const PADDLE_SPEED: usize = 6;
pub const PADDLE_OFFSET: usize = 12;

pub const COLOR: u8 = 0xFF;
pub const WINDOW_SCALE: usize = 1;
pub const BORDER: usize = 1;
pub const TOTAL_WIDTH: usize = WIDTH + BORDER * 2;
pub const TOTAL_HEIGHT: usize = HEIGHT + BORDER * 2;
