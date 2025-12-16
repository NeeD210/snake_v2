import pygame
import time
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from snake_ai.game import SnakeGame

def manual_play(board_size=10, speed=0.15):
    # Initialize Pygame
    pygame.init()
    CELL_SIZE = 40
    screen_size = board_size * CELL_SIZE
    screen = pygame.display.set_mode((screen_size, screen_size + 100)) # Extra space for text
    pygame.display.set_caption("Manual Snake Test")
    font = pygame.font.SysFont(None, 24)
    
    # Game Loop
    game = SnakeGame(board_size=board_size)
    running = True
    
    print("\n--- MANUAL CHECKLIST ---")
    print("1. [ ] Startup: Verify snake has 3 segments (Green Head + 2 Darker Green Body).")
    print("2. [ ] Movement: Try pressing the arrow key OPPOSITE to current direction.")
    print("       - Expectation: Snake ignores it and continues forward.")
    print("       - Failure: Snake turns 180 and dies instantly (Game Over).")
    print("3. [ ] General: Play a bit to ensure valid moves work.")
    print("------------------------\n")
    
    while running:
        # Event Handling
        action = -1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action = 0
                elif event.key == pygame.K_RIGHT: action = 1
                elif event.key == pygame.K_DOWN: action = 2
                elif event.key == pygame.K_LEFT: action = 3
                elif event.key == pygame.K_r: game.reset()
                elif event.key == pygame.K_q: running = False

        # Step
        # If no input, just continue in current direction (simulating frame advance)
        # But wait, step takes an 'action'.
        # In manual play, if no key, we usually send current direction OR nothing.
        # But game.step logic updates direction = action.
        # So we should send current direction if no key press?
        # game.step logic:
        # if action is valid, direction = action.
        # if action invalid (180), ignores it, keeps old direction.
        
        if action == -1:
            action = game.direction # Continue straight
            
        _, reward, done = game.step(action)
        
        # Render
        screen.fill((0, 0, 0)) # Black background
        
        # Draw Play Area
        pygame.draw.rect(screen, (20, 20, 20), (0, 0, screen_size, screen_size))
        
        # Draw Food (Red)
        fx, fy = game.food
        pygame.draw.rect(screen, (255, 0, 0), (fx*CELL_SIZE, fy*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Draw Snake (Green)
        for i, (bx, by) in enumerate(game.snake):
            color = (0, 255, 0)
            if i == 0: color = (100, 255, 100) # Head
            else: color = (0, 200, 0) # Body
            
            pygame.draw.rect(screen, color, (bx*CELL_SIZE, by*CELL_SIZE, CELL_SIZE, CELL_SIZE))
            # Border
            pygame.draw.rect(screen, (0, 100, 0), (bx*CELL_SIZE, by*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

        # Draw UI/Instructions
        text_y = screen_size + 10
        instructions = [
            f"Score: {game.score} | Length: {len(game.snake)}",
            "Controls: Arrow Keys",
            "R: Reset | Q: Quit",
            "CHECK: Try moving backwards!"
        ]
        
        for line in instructions:
            img = font.render(line, True, (255, 255, 255))
            screen.blit(img, (10, text_y))
            text_y += 20

        pygame.display.flip()
        
        if done:
            print(f"Game Over! Score: {game.score}")
            # Show Game Over screen briefly
            screen.fill((50, 0, 0))
            go_text = font.render(f"GAME OVER! Score: {game.score}", True, (255, 255, 255))
            screen.blit(go_text, (screen_size//2 - 100, screen_size//2))
            pygame.display.flip()
            time.sleep(1)
            game.reset()
            
        time.sleep(speed)

    pygame.quit()

if __name__ == "__main__":
    manual_play()
