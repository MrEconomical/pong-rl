mod frame;
mod render;

use crate::config::{
    BALL_SIZE, BALL_SPEED, HEIGHT, MAX_BOUNCE_ANGLE, MAX_INITIAL_ANGLE, PADDLE_HEIGHT,
    PADDLE_OFFSET, PADDLE_SPEED, PADDLE_WIDTH, WIDTH,
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

#[derive(Clone, Copy)]
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
    // Create game with optional Pixels display

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
            Self::move_paddle(&mut self.left_paddle, paddle_move);
        }

        // Update bot paddle position

        let bot_middle = self.right_paddle.1 + PADDLE_HEIGHT / 2;
        let ball_middle = self.ball.1.round() as usize + BALL_SIZE / 2;
        #[allow(clippy::comparison_chain)]
        if ball_middle.abs_diff(bot_middle) >= PADDLE_SPEED {
            if ball_middle < bot_middle {
                Self::move_paddle(&mut self.right_paddle, PaddleMove::Up);
            } else if ball_middle > bot_middle {
                Self::move_paddle(&mut self.right_paddle, PaddleMove::Down);
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

    // Get ball position, ball velocity, and paddle positions

    pub fn get_game_state(&self) -> [f64; 6] {
        [
            self.ball.0,
            self.ball.1,
            self.ball_velocity.x,
            self.ball_velocity.y,
            self.left_paddle.1 as f64,
            self.right_paddle.1 as f64,
        ]
    }

    // Move ball with collision detection and return if game ended

    fn move_ball(&mut self, to: FloatPoint) -> Option<GameResult> {
        // Check for wall and paddle collisions

        if let Some(wall_collision) = self.check_wall_collision(to) {
            return wall_collision;
        }
        let paddle_collision = self.check_paddle_collision(to);
        if paddle_collision {
            return None;
        }

        // Check for left or right wall collision ending game

        if to.0 <= 0.0 {
            return Some(GameResult::Lose);
        } else if to.0 >= (WIDTH - BALL_SIZE) as f64 {
            return Some(GameResult::Win);
        }

        // No obstructions

        self.ball = to;
        None
    }

    // Move paddle in direction considering wall bounds

    fn move_paddle(paddle: &mut Point, direction: PaddleMove) {
        match direction {
            PaddleMove::Up => {
                if paddle.1 <= PADDLE_SPEED {
                    paddle.1 = 0;
                } else {
                    paddle.1 -= PADDLE_SPEED;
                }
            }
            PaddleMove::Down => {
                if paddle.1 >= HEIGHT - PADDLE_HEIGHT - PADDLE_SPEED {
                    paddle.1 = HEIGHT - PADDLE_HEIGHT;
                } else {
                    paddle.1 += PADDLE_SPEED;
                }
            }
        }
    }

    // Check for top or bottom wall collision when moving ball

    fn check_wall_collision(&mut self, to: FloatPoint) -> Option<Option<GameResult>> {
        if to.1 < 0.0 {
            // Move ball to top wall and then move to expected bounce point

            let bounce_dx = self.ball_velocity.x * (to.1 / self.ball_velocity.y);
            let remaining_dx = self.ball_velocity.x - bounce_dx;

            self.ball.0 += bounce_dx;
            self.ball.1 = 0.0;
            self.ball_velocity.y *= -1.0;

            return Some(self.move_ball(FloatPoint(self.ball.0 + remaining_dx, -to.1)));
        } else if to.1 >= (HEIGHT - BALL_SIZE) as f64 {
            // Move ball to bottom wall and then move to expected bounce point

            let extra_dy = to.1 - (HEIGHT - BALL_SIZE) as f64;
            let bounce_dx = self.ball_velocity.x * (extra_dy / self.ball_velocity.y);
            let remaining_dx = self.ball_velocity.x - bounce_dx;

            self.ball.0 += bounce_dx;
            self.ball.1 = (HEIGHT - BALL_SIZE) as f64;
            self.ball_velocity.y *= -1.0;

            return Some(self.move_ball(FloatPoint(
                self.ball.0 + remaining_dx,
                (HEIGHT - BALL_SIZE) as f64 - extra_dy,
            )));
        }

        None
    }

    // Check for left or right paddle collision when moving ball

    fn check_paddle_collision(&mut self, to: FloatPoint) -> bool {
        const LEFT_BOUND: f64 = (PADDLE_OFFSET + PADDLE_WIDTH) as f64;
        const RIGHT_BOUND: f64 = (WIDTH - PADDLE_OFFSET - PADDLE_WIDTH - BALL_SIZE) as f64;
        let prev_x = to.0 - self.ball_velocity.x;

        if to.0 <= LEFT_BOUND
            && prev_x >= LEFT_BOUND
            && to.1 > self.left_paddle.1 as f64 - BALL_SIZE as f64
            && to.1 < self.left_paddle.1 as f64 + PADDLE_HEIGHT as f64
        {
            // Move ball against left paddle

            let extra_dx = LEFT_BOUND - to.0;
            let bounce_dy = self.ball_velocity.y * (extra_dx / self.ball_velocity.x);
            self.ball.0 = LEFT_BOUND;
            self.ball.1 += bounce_dy;

            // Change ball velocity based on bounce position

            let angle = Self::calc_bounce_angle(self.ball.1, self.left_paddle.1);
            self.ball_velocity.x = BALL_SPEED * angle.to_radians().cos();
            self.ball_velocity.y = BALL_SPEED * angle.to_radians().sin();

            return true;
        } else if to.0 >= RIGHT_BOUND
            && prev_x <= RIGHT_BOUND
            && to.1 > self.right_paddle.1 as f64 - BALL_SIZE as f64
            && to.1 < self.right_paddle.1 as f64 + PADDLE_HEIGHT as f64
        {
            // Move ball against right paddle

            let extra_dx = to.0 - RIGHT_BOUND;
            let bounce_dy = self.ball_velocity.y * (extra_dx / self.ball_velocity.x);
            self.ball.0 = RIGHT_BOUND;
            self.ball.1 += bounce_dy;

            // Change ball velocity based on bounce position

            let angle = Self::calc_bounce_angle(self.ball.1, self.right_paddle.1);
            self.ball_velocity.x = -BALL_SPEED * angle.to_radians().cos();
            self.ball_velocity.y = BALL_SPEED * angle.to_radians().sin();

            return true;
        }

        false
    }

    // Calculate ball bounce angle off paddle

    fn calc_bounce_angle(ball_y: f64, paddle_y: usize) -> f64 {
        let ball_center = ball_y + BALL_SIZE as f64 / 2.0;
        let paddle_center = paddle_y as f64 + PADDLE_HEIGHT as f64 / 2.0;
        let real_offset = 2.0 * (ball_center - paddle_center) / PADDLE_HEIGHT as f64;

        const ANGLE_SCALE: f64 = 0.75;
        let scale_factor = if real_offset < 0.0 {
            -(-real_offset).powf(ANGLE_SCALE)
        } else {
            real_offset.powf(ANGLE_SCALE)
        };
        scale_factor * MAX_BOUNCE_ANGLE
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
