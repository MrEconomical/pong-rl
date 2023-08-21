mod frame;
mod render;

use crate::config::{
    BALL_SIZE, BALL_SPEED, HEIGHT, MAX_BALL_ANGLE, MAX_INITIAL_ANGLE, PADDLE_HEIGHT, PADDLE_OFFSET,
    PADDLE_SPEED, PADDLE_WIDTH, WIDTH,
};
use frame::{FloatPoint, Frame, Point};
use std::sync::{Arc, Mutex};

use pixels::Pixels;
use rand::Rng;

// Ball velocity

#[derive(Clone, Copy)]
struct Velocity {
    x: f64,
    y: f64,
}

// Paddle movement input direction

#[derive(Clone, Copy, Debug)]
pub enum PaddleMove {
    Up,
    Down,
}

// User game result

#[derive(Clone, Copy)]
pub enum GameResult {
    Win,
    Lose,
}

// Core Pong game

pub struct Pong {
    ball: FloatPoint,
    ball_velocity: Velocity,
    left_paddle: Point,
    right_paddle: Point,
    frame: Frame,
    ended: bool,
}

impl Pong {
    // Create pong game with optional Pixels display

    pub fn new(pixels: Option<Arc<Mutex<Pixels>>>) -> Self {
        Pong {
            ball: Pong::initial_ball_pos(),
            ball_velocity: Pong::random_initial_velocity(),
            left_paddle: Pong::initial_left_paddle_pos(),
            right_paddle: Pong::initial_right_paddle_pos(),
            frame: Frame::uninit(pixels),
            ended: false,
        }
    }

    // Render initial frame with initial state

    pub fn start_game(&mut self) {
        self.frame
            .init_state(self.ball, self.left_paddle, self.right_paddle);
    }

    // Advance game with user action and return game state

    pub fn tick(&mut self, input: Option<PaddleMove>) -> Option<GameResult> {
        assert!(!self.ended, "cannot run tick after game end");

        // Update player paddle position

        if let Some(paddle_move) = input {
            match paddle_move {
                PaddleMove::Up => {
                    if self.left_paddle.1 <= PADDLE_SPEED {
                        self.left_paddle.1 = 0;
                    } else {
                        self.left_paddle.1 -= PADDLE_SPEED;
                    }
                }
                PaddleMove::Down => {
                    if self.left_paddle.1 >= HEIGHT - PADDLE_HEIGHT - PADDLE_SPEED {
                        self.left_paddle.1 = HEIGHT - PADDLE_HEIGHT;
                    } else {
                        self.left_paddle.1 += PADDLE_SPEED;
                    }
                }
            }
        }

        // Update bot paddle position

        let bot_middle = self.right_paddle.1 + PADDLE_HEIGHT / 2;
        let ball_middle = self.ball.1.round() as usize + BALL_SIZE / 2;
        if ball_middle.abs_diff(bot_middle) >= PADDLE_SPEED {
            if ball_middle < bot_middle {
                if self.right_paddle.1 <= PADDLE_SPEED {
                    self.right_paddle.1 = 0;
                } else {
                    self.right_paddle.1 -= PADDLE_SPEED;
                }
            } else if ball_middle > bot_middle {
                if self.right_paddle.1 >= HEIGHT - PADDLE_HEIGHT - PADDLE_SPEED {
                    self.right_paddle.1 = HEIGHT - PADDLE_HEIGHT;
                } else {
                    self.right_paddle.1 += PADDLE_SPEED;
                }
            }
        }

        // Update ball position

        let game_result = self.move_ball(FloatPoint(
            self.ball.0 + self.ball_velocity.x,
            self.ball.1 + self.ball_velocity.y,
        ));

        // Update frame with new state

        self.frame
            .update(self.ball, self.left_paddle, self.right_paddle);

        if game_result.is_some() {
            self.ended = true;
        }
        game_result
    }

    // Reset game without rerender

    pub fn clear_game(&mut self) {
        self.frame.reset();
        self.ball = Pong::initial_ball_pos();
        self.ball_velocity = Pong::random_initial_velocity();
        self.left_paddle = Pong::initial_left_paddle_pos();
        self.right_paddle = Pong::initial_right_paddle_pos();
        self.ended = false;
    }

    // Move ball with collision detection and return if game ended

    fn move_ball(&mut self, to: FloatPoint) -> Option<GameResult> {
        // Check for top or bottom wall collision

        if to.1 < 0.0 {
            let bounce_dx = self.ball_velocity.x * (to.1 / self.ball_velocity.y);
            let remaining_dx = self.ball_velocity.x - bounce_dx;

            self.ball.0 += bounce_dx;
            self.ball.1 = 0.0;
            self.ball_velocity.y *= -1.0;

            return self.move_ball(FloatPoint(self.ball.0 + remaining_dx, -to.1));
        } else if to.1 >= (HEIGHT - BALL_SIZE) as f64 {
            let extra_dy = to.1 - (HEIGHT - BALL_SIZE) as f64;
            let bounce_dx = self.ball_velocity.x * (extra_dy / self.ball_velocity.y);
            let remaining_dx = self.ball_velocity.x - bounce_dx;

            self.ball.0 += bounce_dx;
            self.ball.1 = (HEIGHT - BALL_SIZE) as f64;
            self.ball_velocity.y *= -1.0;

            return self.move_ball(FloatPoint(
                self.ball.0 + remaining_dx,
                (HEIGHT - BALL_SIZE) as f64 - extra_dy,
            ));
        }

        // Check for paddle collision

        const LEFT_BOUND: f64 = (PADDLE_OFFSET + PADDLE_WIDTH) as f64;
        const RIGHT_BOUND: f64 = (WIDTH - PADDLE_OFFSET - PADDLE_WIDTH - BALL_SIZE) as f64;

        if to.0 <= LEFT_BOUND {}

        self.ball = to;
        None
    }

    // Return initial game values

    fn initial_ball_pos() -> FloatPoint {
        FloatPoint(
            (WIDTH / 2 - BALL_SIZE / 2) as f64,
            (HEIGHT / 2 - BALL_SIZE / 2) as f64,
        )
    }

    fn random_initial_velocity() -> Velocity {
        let angle = rand::thread_rng().gen_range(-MAX_INITIAL_ANGLE..MAX_INITIAL_ANGLE);
        Velocity {
            x: BALL_SPEED * angle.to_radians().cos(),
            y: BALL_SPEED * angle.to_radians().sin(),
        }
    }

    fn initial_left_paddle_pos() -> Point {
        Point(PADDLE_OFFSET, HEIGHT / 2 - PADDLE_HEIGHT / 2)
    }

    fn initial_right_paddle_pos() -> Point {
        Point(
            WIDTH - PADDLE_OFFSET - PADDLE_WIDTH,
            HEIGHT / 2 - PADDLE_HEIGHT / 2,
        )
    }
}
