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

// Ball velocity struct

#[derive(Clone, Copy)]
struct Velocity {
    x: f64,
    y: f64,
}

// Paddle movement input direction enum

#[derive(Clone, Copy)]
pub enum PaddleMove {
    Up,
    Down,
}

// User game result enum

#[derive(Clone, Copy)]
pub enum GameResult {
    Win,
    Lose,
}

// Core Pong game struct

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

        // Update ball position

        let game_result = self.move_ball(FloatPoint(
            self.ball.0 + self.ball_velocity.x,
            self.ball.1 + self.ball_velocity.y,
        ));

        // Update frame with new state

        self.frame
            .update(self.ball, self.left_paddle, self.right_paddle);

        if let Some(_) = game_result {
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
        println!("moving ball to {} {}", to.0, to.1);

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
