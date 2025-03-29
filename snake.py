import pygame, sys, random
import numpy as np
from enum import Enum

class Direction(Enum):
    RIGHT = (0, 1)
    DOWN = (1, 0)
    LEFT = (0, -1)
    UP = (-1, 0)

class SnakeGame:
    def __init__(self, width=20, height=20, cell_size=20, fps=10, seed=None):
        # Game parameters
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.fps = fps
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Game state variables
        self.snake = []
        self.direction = Direction.RIGHT.value
        self.food_position = None
        self.score = 0
        self.game_over = False
        self.steps = 0
        
        # Initialize game
        self.reset()
        
        # Set up pygame for human play (can be disabled for RL training)
        self.display_game = True
        if self.display_game:
            pygame.init()
            self.screen_width = width * cell_size
            self.screen_height = height * cell_size
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Snake Game')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 25)
    
    def reset(self):
        """Reset the game state for a new episode."""
        # Start with a snake of length 3 in the middle of the board
        middle_x, middle_y = self.width // 2, self.height // 2
        self.snake = [(middle_y, middle_x), 
                      (middle_y, middle_x - 1), 
                      (middle_y, middle_x - 2)]
        self.direction = Direction.RIGHT.value
        self.place_food()
        self.score = 0
        self.game_over = False
        self.steps = 0
        return self.get_state()
    
    def place_food(self):
        """Place food in a random empty cell."""
        empty_cells = [
            (i, j) for i in range(self.height) for j in range(self.width)
            if (i, j) not in self.snake
        ]
        if empty_cells:
            self.food_position = random.choice(empty_cells)
    
    def update(self, direction=None):
        """Update game state based on direction."""
        # If direction is provided and valid, update direction
        if direction and self.is_valid_direction(direction):
            self.direction = direction
        
        # Calculate new head position
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Check for collisions with walls
        if (new_head[0] < 0 or new_head[0] >= self.height or
            new_head[1] < 0 or new_head[1] >= self.width):
            self.game_over = True
            return
        
        # Check for collisions with self
        if new_head in self.snake:
            self.game_over = True
            return
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check if food eaten
        if new_head == self.food_position:
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()  # Remove tail if no food eaten
        
        self.steps += 1
    
    def is_valid_direction(self, new_dir):
        """Prevent 180-degree turns."""
        return not (new_dir[0] == -self.direction[0] and 
                    new_dir[1] == -self.direction[1])
    
    def get_state(self):
        """Convert game state to a representation suitable for DQN."""
        # Create a grid representation with channels for snake body, head, and food
        state = np.zeros((self.height, self.width, 3))
        
        # Mark snake body
        for segment in self.snake[1:]:
            state[segment[0], segment[1], 0] = 1
        
        # Mark snake head
        head = self.snake[0]
        state[head[0], head[1], 1] = 1
        
        # Mark food
        state[self.food_position[0], self.food_position[1], 2] = 1
        
        return state
    
    def get_danger_state(self):
        """Alternative state representation focused on danger and food direction."""
        head = self.snake[0]
        
        # Check danger in each direction
        danger_straight = self.check_danger(self.direction)
        danger_right = self.check_danger(self.get_right_direction())
        danger_left = self.check_danger(self.get_left_direction())
        
        # Determine food direction relative to head
        food_dir = (
            self.food_position[0] - head[0],
            self.food_position[1] - head[1]
        )
        
        # Food direction flags
        food_up = food_dir[0] < 0
        food_right = food_dir[1] > 0
        food_down = food_dir[0] > 0
        food_left = food_dir[1] < 0
        
        # Current direction flags
        dir_right = self.direction == Direction.RIGHT.value
        dir_down = self.direction == Direction.DOWN.value
        dir_left = self.direction == Direction.LEFT.value
        dir_up = self.direction == Direction.UP.value
        
        # Return state as vector
        return np.array([
            # Danger
            danger_straight,
            danger_right,
            danger_left,
            
            # Current direction
            dir_right,
            dir_down,
            dir_left,
            dir_up,
            
            # Food direction
            food_up,
            food_right,
            food_down,
            food_left
        ], dtype=np.float32)
    
    def check_danger(self, direction):
        """Check if moving in a direction leads to danger."""
        head = self.snake[0]
        new_pos = (head[0] + direction[0], head[1] + direction[1])
        
        # Check wall collision
        if (new_pos[0] < 0 or new_pos[0] >= self.height or
            new_pos[1] < 0 or new_pos[1] >= self.width):
            return True
        
        # Check self collision
        if new_pos in self.snake:
            return True
        
        return False
    
    def get_right_direction(self):
        """Get the direction to the right of current direction."""
        if self.direction == Direction.RIGHT.value:
            return Direction.DOWN.value
        if self.direction == Direction.DOWN.value:
            return Direction.LEFT.value
        if self.direction == Direction.LEFT.value:
            return Direction.UP.value
        return Direction.RIGHT.value
    
    def get_left_direction(self):
        """Get the direction to the left of current direction."""
        if self.direction == Direction.RIGHT.value:
            return Direction.UP.value
        if self.direction == Direction.UP.value:
            return Direction.LEFT.value
        if self.direction == Direction.LEFT.value:
            return Direction.DOWN.value
        return Direction.RIGHT.value
    
    def handle_events(self):
        """Handle keyboard events for human play."""
        new_direction = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT and self.direction != Direction.LEFT.value:
                    new_direction = Direction.RIGHT.value
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP.value:
                    new_direction = Direction.DOWN.value
                elif event.key == pygame.K_LEFT and self.direction != Direction.RIGHT.value:
                    new_direction = Direction.LEFT.value
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN.value:
                    new_direction = Direction.UP.value
        
        # Only update direction once per game cycle
        if new_direction:
            self.direction = new_direction
    
    def render(self):
        """Render the game for human play."""
        if not self.display_game:
            return
            
        # Clear the screen
        self.screen.fill((0, 0, 0))
        
        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), 
                            (segment[1] * self.cell_size, segment[0] * self.cell_size, 
                            self.cell_size, self.cell_size))
        
        # Draw food
        pygame.draw.rect(self.screen, (255, 0, 0), 
                        (self.food_position[1] * self.cell_size, self.food_position[0] * self.cell_size, 
                        self.cell_size, self.cell_size))
        
        # Draw score
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (5, 5))
        
        # Update display
        pygame.display.update()
        self.clock.tick(self.fps)
    
    def play_human(self):
        """Main game loop for human play."""
        while not self.game_over:
            self.handle_events()
            self.update()
            self.render()
        
        # Display game over
        game_over_text = self.font.render('Game Over! Press any key to exit.', True, (255, 255, 255))
        self.screen.blit(game_over_text, (self.screen_width // 4, self.screen_height // 2))
        pygame.display.update()
        
        # Wait for key press to exit
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False
        
        pygame.quit()

# Run the game if this file is executed directly
if __name__ == "__main__":
    game = SnakeGame(width=30, height=30, cell_size=25)
    game.play_human()