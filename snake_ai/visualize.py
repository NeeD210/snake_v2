import pygame
import torch
import numpy as np
import time
import torch.nn.functional as F
from game import SnakeGame
from model import SnakeNet
from mcts import MCTS

def get_pov_state(game, state):
    """
    Process state to match training POV (Head Up).
    """
    input_tensor = np.zeros((4, game.board_size, game.board_size), dtype=np.float32)

    # Channel 0: Lifetime/flow of the body (temporal information).
    snake = getattr(game, "snake", [])
    L = len(snake)
    if L > 1:
        for i in range(1, L):
            x, y = snake[i]
            input_tensor[0, y, x] = (L - i) / L

    input_tensor[1] = (state == 2).astype(float) # Head
    input_tensor[2] = (state == 3).astype(float) # Food
    hunger_limit = max(1, getattr(game, "hunger_limit", 100))
    hunger = float(getattr(game, "steps_since_eaten", 0)) / hunger_limit
    input_tensor[3].fill(hunger)
    
    # Rotate based on direction to enforce POV (Head Up)
    # k=0 (Up) -> 0 rot
    # k=1 (Right) -> 1 rot (90 deg CCW) -> Right becomes Up
    k = game.direction
    input_tensor = np.rot90(input_tensor, k, axes=(1, 2)).copy()
    
    return input_tensor

def draw_pov(screen, pov_tensor, start_x, start_y, cell_size):
    """
    Draws the POV Grid.
    pov_tensor shape: (3, H, W)
    """
    channels, H, W = pov_tensor.shape
    
    # Border
    pygame.draw.rect(screen, (255, 255, 255), (start_x - 2, start_y - 2, W*cell_size + 4, H*cell_size + 4), 1)
    
    for y in range(H):
        for x in range(W):
            rect = (start_x + x*cell_size, start_y + y*cell_size, cell_size, cell_size)
            
            # Determine color
            # Channel 0: Body (Green)
            # Channel 1: Head (Blue/Cyan)
            # Channel 2: Food (Red)
            # Channel 3: Hunger (not drawn)
            
            color = (50, 50, 50) # Empty (Dark Grey)
            if pov_tensor[1, y, x] > 0: color = (0, 255, 255) # Head
            elif pov_tensor[0, y, x] > 0: color = (0, 150, 0) # Body
            elif pov_tensor[2, y, x] > 0: color = (255, 0, 0) # Food
            
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (30, 30, 30), rect, 1) # Grid lines

def visualize(model_path="snake_net.pth", board_size=10, speed=0.1, debug_inputs=False, use_mcts=False, simulations=50):
    # Initialize Pygame
    pygame.init()
    CELL_SIZE = 40
    
    # Layout dimensions
    GAME_SIZE = board_size * CELL_SIZE
    INFO_WIDTH = 300
    SCREEN_WIDTH = GAME_SIZE + INFO_WIDTH
    SCREEN_HEIGHT = max(GAME_SIZE, 400) # Min height for info
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("MCTS Snake AI - POV Analysis")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    large_font = pygame.font.SysFont("Arial", 24)

    # Load Model
    model = SnakeNet(board_size=board_size)
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print("Model file not found! Please run main.py to train first.")
        return
    except RuntimeError as e:
        print("Model file exists, but weights are incompatible with the current architecture.")
        print("If you recently changed input features/channels, retrain a new model.")
        print(f"Details: {e}")
        return

    model.eval()
    
    
    # Initialize MCTS if enabled
    mcts = None
    if use_mcts:
        print(f"Initializing MCTS with {simulations} simulations...")
        mcts = MCTS(model, n_simulations=simulations)
        mcts.reset()

    # Game Loop
    game = SnakeGame(board_size=board_size)
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_q: running = False

        # AI Prediction
        state_grid = game.get_state()
        pov_numpy = get_pov_state(game, state_grid) # (3, H, W)
        
        if debug_inputs:
            print(f"\n--- Step Info (Dir: {game.direction}) ---")
            print("Channel 0: Body")
            print(pov_numpy[0])
            print("Channel 1: Head")
            print(pov_numpy[1])
            print("Channel 2: Food")
            print(pov_numpy[2])
            print("-" * 30)

        if use_mcts:
            # MCTS Prediction
            p_probs, entropy = mcts.search(game)
            # Value is Q of the chosen action or max Q?
            # Or root value?
            if mcts.root:
                value = mcts.root.Q
            else:
                value = 0.0
            
            # Choose best action (Greedy based on visits)
            rel_action = np.argmax(p_probs)
            abs_action = (game.direction + (rel_action - 1)) % 4
            
        else:
            # Raw NN Prediction
            input_tensor = torch.tensor(pov_numpy).unsqueeze(0) # (1, 3, H, W)
    
            with torch.no_grad():
                p_logits, v = model(input_tensor)
                
                # Convert logits to probs
                p_probs = torch.exp(p_logits).cpu().numpy()[0] # [Left, Straight, Right]
                value = v.item()
                
                # Choose best action (Greedy)
                rel_action = np.argmax(p_probs)
                
                # Convert to absolute
                # 0: Left, 1: Straight, 2: Right
                abs_action = (game.direction + (rel_action - 1)) % 4

        # --- RENDER BEFORE STEPPING (Syncs POV with Game State) ---
        screen.fill((20, 20, 20)) # Dark background
        
        # 1. Draw Game Board (Left Side)
        pygame.draw.rect(screen, (0, 0, 0), (0, 0, GAME_SIZE, GAME_SIZE))
        
        # Draw Food
        fx, fy = game.food
        pygame.draw.rect(screen, (255, 50, 50), (fx*CELL_SIZE, fy*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Draw Snake
        for i, (bx, by) in enumerate(game.snake):
            color = (0, 255, 0)
            if i == 0: color = (150, 255, 150) # Head lighter
            pygame.draw.rect(screen, color, (bx*CELL_SIZE, by*CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, (0, 100, 0), (bx*CELL_SIZE, by*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

        # 2. Draw Speed Lines/Direction
        # Maybe an arrow on the head?
        head_x, head_y = game.snake[0]
        center_x = head_x * CELL_SIZE + CELL_SIZE // 2
        center_y = head_y * CELL_SIZE + CELL_SIZE // 2
        # Dir: 0:Up, 1:Right, 2:Down, 3:Left
        end_x, end_y = center_x, center_y
        if game.direction == 0: end_y -= 15
        elif game.direction == 1: end_x += 15
        elif game.direction == 2: end_y += 15
        elif game.direction == 3: end_x -= 15
        pygame.draw.line(screen, (0, 0, 255), (center_x, center_y), (end_x, end_y), 3)

        # 3. Draw Info Panel (Right Side)
        panel_x = GAME_SIZE + 10
        y_offset = 10
        
        # Title
        # Title
        title_text = "MCTS Agent State" if use_mcts else "Neural Internal State"
        title = large_font.render(title_text, True, (255, 255, 255))
        screen.blit(title, (panel_x, y_offset))
        y_offset += 40
        
        # Value Estimate
        v_text = font.render(f"Value Est: {value:.3f}", True, (200, 200, 200))
        screen.blit(v_text, (panel_x, y_offset))
        y_offset += 30
        
        # POV Visualization
        pov_text = font.render("Snake's Eye View (POV):", True, (200, 200, 200))
        screen.blit(pov_text, (panel_x, y_offset))
        y_offset += 25
        
        # Draw small POV grid
        POV_CELL_SIZE = 15
        draw_pov(screen, pov_numpy, panel_x, y_offset, POV_CELL_SIZE)
        y_offset += board_size * POV_CELL_SIZE + 20
        
        # Action Probabilities
        prob_text = font.render("Action Probabilities:", True, (200, 200, 200))
        screen.blit(prob_text, (panel_x, y_offset))
        y_offset += 25
        
        actions = ["Left", "Straight", "Right"]
        bar_width = 150
        bar_height = 20
        
        for i, prob in enumerate(p_probs):
            action_name = actions[i]
            # Highlight chosen action
            text_color = (255, 255, 255) if i == rel_action else (150, 150, 150)
            label = font.render(f"{action_name}: {prob:.2f}", True, text_color)
            screen.blit(label, (panel_x, y_offset))
            
            # Bar
            pygame.draw.rect(screen, (50, 50, 50), (panel_x + 90, y_offset + 2, bar_width, bar_height))
            fill_width = int(bar_width * prob)
            bar_color = (0, 255, 0) if i == rel_action else (100, 100, 100)
            pygame.draw.rect(screen, bar_color, (panel_x + 90, y_offset + 2, fill_width, bar_height))
            
            y_offset += 30

        pygame.display.flip()
        
        # Step
        _, reward, done = game.step(abs_action)
        
        if use_mcts:
            mcts.update_root(rel_action)
        
        if done:
            print(f"Game Over! Score: {game.score} | Reason: {game.death_reason}")
            print(f"Last Head: {game.snake[0]}, Dir: {game.direction}, Action: {actions[rel_action]} ({abs_action})")
            time.sleep(1)
            time.sleep(1)
            game.reset()
            if use_mcts: mcts.reset()
            
        time.sleep(speed)

    pygame.quit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="snake_ai/experiments/train_v4/snake_net.pth", help="Path to model file")
    parser.add_argument("--board_size", type=int, default=6, help="Board size")
    parser.add_argument("--speed", "-s", type=float, default=0.2, help="Game speed (seconds per frame)")
    parser.add_argument("--debug-inputs", action="store_true", help="Print input tensors to console for debugging")
    
    parser.add_argument("--mcts", action="store_true", help="Use MCTS for decision making")
    parser.add_argument("--sims", type=int, default=50, help="Number of MCTS simulations")
    
    args = parser.parse_args()
    
    visualize(model_path=args.model, board_size=args.board_size, speed=args.speed, debug_inputs=args.debug_inputs, use_mcts=args.mcts, simulations=args.sims)
