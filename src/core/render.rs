use super::frame::{FloatPoint, Point};
use crate::config::{BALL_SIZE, BORDER, COLOR, HEIGHT, TOTAL_HEIGHT, TOTAL_WIDTH};

// Draw ball on Pixels RGBA frame at position with subpixel rendering

pub fn draw_ball(rgba_frame: &mut [u8], pos: FloatPoint, color: u8) {
    let (start_pos, ball_pixels) = calc_ball_pixels(pos, color);
    for row in 0..BALL_SIZE + 1 {
        let row_offset = start_pos.1 + row + BORDER;
        if row_offset == TOTAL_HEIGHT - BORDER {
            continue;
        }
        for col in 0..BALL_SIZE + 1 {
            let offset = (row_offset * TOTAL_WIDTH + start_pos.0 + col + BORDER) * 4;
            set_rgba_color(rgba_frame, offset, ball_pixels[row][col]);
        }
    }
}

// Draw rectangle on Pixels RGBA frame at position with width and height

pub fn draw_rect(rgba_frame: &mut [u8], pos: Point, width: usize, height: usize, color: u8) {
    for row in pos.1..pos.1 + height {
        for col in pos.0..pos.0 + width {
            let offset = ((row + BORDER) * TOTAL_WIDTH + col + BORDER) * 4;
            set_rgba_color(rgba_frame, offset, color);
        }
    }
}

// Draw border on Pixels RGBA frame

pub fn draw_border(rgba_frame: &mut [u8]) {
    for row in 0..BORDER {
        for col in 0..TOTAL_WIDTH {
            let offset_top = (row * TOTAL_WIDTH + col) * 4;
            set_rgba_color(rgba_frame, offset_top, COLOR);
            let offset_bottom = ((TOTAL_HEIGHT - row - 1) * TOTAL_WIDTH + col) * 4;
            set_rgba_color(rgba_frame, offset_bottom, COLOR);
        }
    }

    for col in 0..BORDER {
        for row in 0..HEIGHT {
            let offset_left = ((row + BORDER) * TOTAL_WIDTH + col) * 4;
            set_rgba_color(rgba_frame, offset_left, COLOR);
            let offset_right = ((row + BORDER) * TOTAL_WIDTH + TOTAL_WIDTH - col - 1) * 4;
            set_rgba_color(rgba_frame, offset_right, COLOR);
        }
    }
}

// Calculate pixel colors for ball subpixel rendering

fn calc_ball_pixels(pos: FloatPoint, color: u8) -> (Point, [[u8; BALL_SIZE + 1]; BALL_SIZE + 1]) {
    let start_pos = Point(pos.0.floor() as usize, pos.1.floor() as usize);
    let mut ball_pixels = [[0x00; BALL_SIZE + 1]; BALL_SIZE + 1];
    if color == 0x00 {
        return (start_pos, ball_pixels);
    }

    let left_bound = pos.0;
    let right_bound = pos.0 + (BALL_SIZE - 1) as f64;
    let top_bound = pos.1;
    let bottom_bound = pos.1 + (BALL_SIZE - 1) as f64;

    for row in 0..BALL_SIZE + 1 {
        for col in 0..BALL_SIZE + 1 {
            // Calculate intersect area for color intensity

            let x = (start_pos.0 + col) as f64;
            let width = if x < left_bound {
                1.0 - (left_bound - x)
            } else if x > right_bound {
                1.0 - (x - right_bound)
            } else {
                1.0
            };

            let y = (start_pos.1 + row) as f64;
            let height = if y < top_bound {
                1.0 - (top_bound - y)
            } else if y > bottom_bound {
                1.0 - (y - bottom_bound)
            } else {
                1.0
            };

            // Scale color intensity and discard dim colors

            const MIN_COLOR: u8 = COLOR / 2;
            let square_color = (width * height * color as f64).round() as u8;
            ball_pixels[row][col] = if square_color >= MIN_COLOR {
                square_color
            } else {
                0x00
            };
        }
    }

    (start_pos, ball_pixels)
}

// Set RGBA color at offset in frame with zero transparency

fn set_rgba_color(frame: &mut [u8], offset: usize, color: u8) {
    frame[offset] = color;
    frame[offset + 1] = color;
    frame[offset + 2] = color;
    frame[offset + 3] = 0xFF;
}