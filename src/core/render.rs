use super::frame::{FloatPoint, Point};
use crate::config::{BALL_SIZE, WIDTH};

// Draw ball on internal frame at position with subpixel rendering

pub fn draw_ball_internal(frame: &mut [u8], pos: FloatPoint, color: u8) {
    let (start_pos, ball_pixels) = calc_ball_pixels(pos, color);
    for row in 0..BALL_SIZE + 1 {
        for col in 0..BALL_SIZE + 1 {
            let offset = (start_pos.1 + row) * WIDTH + start_pos.0 + col;
            frame[offset] = ball_pixels[row][col];
        }
    }
}

// Draw ball on Pixels RGBA frame at position with subpixel rendering

pub fn draw_ball_display(rgba_frame: &mut [u8], pos: FloatPoint, color: u8) {
    let (start_pos, ball_pixels) = calc_ball_pixels(pos, color);
    for row in 0..BALL_SIZE + 1 {
        for col in 0..BALL_SIZE + 1 {
            let offset = ((start_pos.1 + row) * WIDTH + start_pos.0 + col) * 4;
            rgba_frame[offset] = ball_pixels[row][col];
            rgba_frame[offset + 1] = ball_pixels[row][col];
            rgba_frame[offset + 2] = ball_pixels[row][col];
            rgba_frame[offset + 3] = 0xFF;
        }
    }
}

// Draw rectangle on internal frame at position with width and height

pub fn draw_internal(frame: &mut [u8], pos: Point, width: usize, height: usize, color: u8) {
    for row in pos.1..pos.1 + height {
        for col in pos.0..pos.0 + width {
            frame[row * WIDTH + col] = color;
        }
    }
}

// Draw rectangle on Pixels RGBA frame at position with width and height

pub fn draw_display(rgba_frame: &mut [u8], pos: Point, width: usize, height: usize, color: u8) {
    for row in pos.1..pos.1 + height {
        for col in pos.0..pos.0 + width {
            let offset = (row * WIDTH + col) * 4;
            rgba_frame[offset] = color;
            rgba_frame[offset + 1] = color;
            rgba_frame[offset + 2] = color;
            rgba_frame[offset + 3] = 0xFF;
        }
    }
}

// Calculate pixel colors for ball subpixel rendering

fn calc_ball_pixels(pos: FloatPoint, color: u8) -> (Point, [[u8; BALL_SIZE + 1]; BALL_SIZE + 1]) {
    let start_pos = Point(pos.0.floor() as usize, pos.1.floor() as usize);
    (start_pos, [[color; BALL_SIZE + 1]; BALL_SIZE + 1])
}
