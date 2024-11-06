from game_lib.game_utilities.helper_functions import  *
import pygame
import time
import random
import  numpy as np

class SnakeGame:
    def __init__(self, w=400, h=400):
        """
        Initializes the SnakeGame object.

        Parameters:
        - w (int): Width of the game window.
        - h (int): Height of the game window.
        """
        self.w = w
        self.h = h

        self.background_image = pygame.transform.scale(pygame.image.load("game_lib/game_images/img.jpg"), (self.w, self.h))

        # Initialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.total_time = 0
        self.clock = pygame.time.Clock()
        self.start_time = None

        self.reset()
        self.colors = [
            # "black", #to fill the screen
            "white",
            "red",
            "green",
            "blue",
            "cyan",
            "magenta",
            "yellow",
            "orange",
            "gray"
        ]

    # Reset the game state
    def reset(self):
        """
        Resets the game state to start a new game.
        """
        # Begin with right direction
        self.start_time = time.time()
        self.direction = Direction.RIGHT
        # Start at the middle of the screen
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.frame_iteration = 0
        self.food = None
        self._place_food()

    def get_state(self):
        """
        Get the current state of the game.

        Returns:
        - np.array: Array representing the current state.
        """
        head = self.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def _place_food(self):
        """
        Places the food in a random location on the screen.

        The food should not overlap with the snake.

        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """
        Executes a single step in the game.

        Args:
            action (int): The action to be taken by the agent.

        Returns:
            tuple: A tuple containing reward, game_over flag, current score, and total time played.

        """
        # 1. Collect user input
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move
        self._move(action)  # Update the head
        self.snake.insert(0, self.head)

        reward = 0

        # 3. Check if game over
        game_over = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            # We need to add how much we played until death
            total_time_played = time.time() - self.start_time
            game_over = True
            reward = -10
            return reward, game_over, self.score, total_time_played

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. Return game over, score, and total time played
        total_time_played = time.time() - self.start_time
        return reward, game_over, self.score, total_time_played

    def is_collision(self, pt=None):
        """
        Checks if there is a collision.

        Args:
            pt (Point, optional): The point to check for collision. Defaults to None.

        Returns:
            bool: True if there is a collision, False otherwise.

        """
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def grid(self):
        """
        Draws a grid on the display.

        """
        for row in range(0, self.h, BLOCK_SIZE):
            for col in range(0, self.h, BLOCK_SIZE):
                # Draw rect
                rect = pygame.Rect(row, col, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(self.display, "green", rect, 3)
        pygame.display.update()

    def _update_ui(self):
        """
        Updates the UI elements on the display.

        """
        snake_color = random.choice(self.colors)

        self.display.fill(BLACK)

        # self.grid()

        # add background images to the background
        # self.display.blit(self.background_image, (0, 0))

        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, 'Red', (x, 0), (x, self.h), 1)
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, 'Red', (0, y), (self.w, y), 1)

        for pt in self.snake:
            pygame.draw.circle(self.display, snake_color, (pt.x + BLOCK_SIZE // 2, pt.y + BLOCK_SIZE // 2),
                               BLOCK_SIZE // 2)
            pygame.draw.circle(self.display, snake_color, (pt.x + BLOCK_SIZE // 2, pt.y + BLOCK_SIZE // 2),
                               BLOCK_SIZE // 4)

        pygame.draw.circle(self.display, 'Green', (self.food.x + BLOCK_SIZE // 2, self.food.y + BLOCK_SIZE // 2),
                           BLOCK_SIZE // 2)

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """
        Move the snake based on the action provided.

        Parameters:
        - action: List of three elements representing the action [straight, right, left].

        """
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

