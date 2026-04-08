import pygame
import torch
import numpy as np
import time
import torch.nn.functional as F
from game import SnakeGame
from model import SnakeNet
from mcts import MCTS
from encoder import encode_pov
from schedules import get_mcts_simulations


def _extract_state_dict(ckpt_obj):
    """
    Accept either a raw state_dict or a checkpoint dict.
    """
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
    return ckpt_obj


def _load_model_state_compat(model: torch.nn.Module, model_path: str) -> None:
    """
    Backward-compatible checkpoint loader.

    Supports older checkpoints that used BatchNorm layer names (bn1/bn2/bn3)
    when the current architecture uses GroupNorm (gn1/gn2/gn3).
    """
    ckpt_obj = torch.load(model_path, map_location="cpu")
    src_sd = _extract_state_dict(ckpt_obj)
    if not isinstance(src_sd, dict):
        raise RuntimeError(f"Unsupported checkpoint format: expected state_dict, got {type(src_sd)}")

    # Common checkpoint format: DataParallel prefixes keys with "module."
    if any(k.startswith("module.") for k in src_sd.keys()):
        src_sd = {k[len("module.") :]: v for k, v in src_sd.items() if isinstance(k, str)}

    dst_sd = model.state_dict()
    merged = {}
    loaded_keys = set()

    def maybe_take(dst_key: str, *src_keys: str) -> bool:
        for src_key in src_keys:
            if src_key not in src_sd:
                continue
            if src_sd[src_key].shape != dst_sd[dst_key].shape:
                continue
            merged[dst_key] = src_sd[src_key]
            loaded_keys.add(dst_key)
            return True
        return False

    for k in dst_sd.keys():
        # Direct key match
        if maybe_take(k, k):
            continue

        # conv1 -> input_conv rename compatibility (older checkpoints)
        if k == "input_conv.weight":
            if maybe_take(k, "conv1.weight"):
                continue
        if k == "input_conv.bias":
            if maybe_take(k, "conv1.bias"):
                continue

        # BatchNorm -> GroupNorm rename compatibility
        # (running_mean/var keys are intentionally ignored; GN doesn't have them)
        if (k.endswith(".weight") or k.endswith(".bias")):
            # input_gn.* used to be input_bn.* or bn1.*
            if k.startswith("input_gn."):
                suf = k[len("input_gn.") :]
                if maybe_take(k, f"input_bn.{suf}", f"bn1.{suf}"):
                    continue

            # Residual blocks: blocks.N.gn1/gn2.* used to be blocks.N.bn1/bn2.*
            if ".gn1." in k or ".gn2." in k:
                bn_k = k.replace(".gn1.", ".bn1.").replace(".gn2.", ".bn2.")
                if maybe_take(k, bn_k):
                    continue

        # Default: keep model's initialized weights for missing/unmatched keys
        merged[k] = dst_sd[k]

    # If we couldn't load the input conv, it's usually an input-channel change (e.g. 3->4).
    in_key = "input_conv.weight"
    if in_key in dst_sd and (in_key not in loaded_keys):
        # Check common legacy key names for a shape mismatch to produce a clearer error.
        legacy_keys = ("input_conv.weight", "conv1.weight")
        for lk in legacy_keys:
            if lk in src_sd and src_sd[lk].shape != dst_sd[in_key].shape:
                raise RuntimeError(
                    "Checkpoint input_conv shape mismatch (likely input channels/features changed). "
                    f"checkpoint[{lk}]: {tuple(src_sd[lk].shape)} vs current[{in_key}]: {tuple(dst_sd[in_key].shape)}"
                )

    model.load_state_dict(merged, strict=True)

def get_pov_state(game, state):
    """
    Process state to match training POV (Head Up).
    """
    return encode_pov(game, state)

def draw_pov(screen, pov_tensor, start_x, start_y, cell_size):
    """
    Draws the POV Grid.
    pov_tensor shape: (C, H, W)
    """
    channels, H, W = pov_tensor.shape
    
    # Background for POV
    pygame.draw.rect(screen, (30, 30, 40), (start_x, start_y, W*cell_size, H*cell_size), border_radius=4)
    
    for y in range(H):
        for x in range(W):
            rect = (start_x + x*cell_size, start_y + y*cell_size, cell_size, cell_size)
            
            # Determine color
            if pov_tensor[2, y, x] > 0:
                color = (255, 80, 80) # Food
            elif pov_tensor[1, y, x] > 0:
                color = (0, 255, 150) # Head
            elif pov_tensor[0, y, x] > 0:
                color = (0, 180, 100) # Body
            else:
                continue # Skip empty
            
            pygame.draw.rect(screen, color, rect, border_radius=2)

def draw_snake_segment(screen, x, y, size, color, is_head=False, direction=0, is_tail=False, next_pos=None, prev_pos=None):
    """Draws a snake segment with rounded corners and eyes for the head."""
    rect = pygame.Rect(x * size + 1, y * size + 1, size - 2, size - 2)
    
    # Calculate border radius based on neighbors to make it look connected
    # For now, let's keep it simple with a fixed radius but rounded
    radius = size // 3
    pygame.draw.rect(screen, color, rect, border_radius=radius)
    
    if is_head:
        # Draw eyes
        eye_color = (255, 255, 255)
        pupil_color = (0, 0, 0)
        eye_size = size // 6
        pupil_size = size // 12
        
        # Eye positions based on direction (0:Up, 1:Right, 2:Down, 3:Left)
        offsets = [
            [(size//4, size//4), (3*size//4, size//4)],       # Up
            [(3*size//4, size//4), (3*size//4, 3*size//4)],   # Right
            [(size//4, 3*size//4), (3*size//4, 3*size//4)],   # Down
            [(size//4, size//4), (size//4, 3*size//4)],       # Left
        ]
        
        for ex, ey in offsets[direction]:
            # Adjust to absolute screen coords
            ex_abs = x * size + ex
            ey_abs = y * size + ey
            pygame.draw.circle(screen, eye_color, (ex_abs, ey_abs), eye_size)
            pygame.draw.circle(screen, pupil_color, (ex_abs, ey_abs), pupil_size)

def draw_food(screen, x, y, size, color, pulse_factor=1.0):
    """Draws food as a shiny circle with a pulse effect."""
    center = (x * size + size // 2, y * size + size // 2)
    radius = (size // 2 - 4) * pulse_factor
    
    # Main body
    pygame.draw.circle(screen, color, center, int(radius))
    # Shine
    shine_pos = (center[0] - radius // 3, center[1] - radius // 3)
    pygame.draw.circle(screen, (255, 255, 255), (int(shine_pos[0]), int(shine_pos[1])), int(radius // 4))
    
def visualize(
    model_path="snake_net.pth",
    board_size=10,
    speed=0.1,
    debug_inputs=False,
    use_mcts=False,
    simulations=50,
    *,
    generation: int = 0,
    use_sim_schedule: bool = True,
    dev_mode: bool = False,
):
    # Colors
    BG_DARK = (15, 15, 25)
    BOARD_BG = (25, 25, 35)
    GRID_COLOR = (35, 35, 45)
    TEXT_COLOR = (220, 220, 230)
    SNAKE_HEAD_COLOR = (0, 255, 127)
    SNAKE_BODY_COLOR = (0, 200, 100)
    FOOD_COLOR = (255, 65, 54)
    ACCENT_COLOR = (100, 150, 255)

    # Initialize Pygame
    pygame.init()
    CELL_SIZE = 40
    
    # Layout dimensions
    GAME_SIZE = board_size * CELL_SIZE
    INFO_WIDTH = 350
    SCREEN_WIDTH = GAME_SIZE + INFO_WIDTH
    SCREEN_HEIGHT = max(GAME_SIZE, 500)
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("🐍 Pro Snake AI - " + ("MCTS Mode" if use_mcts else "Neural Mode"))
    clock = pygame.time.Clock()
    
    # Load fonts
    try:
        font_main = pygame.font.SysFont("Segoe UI", 18)
        font_bold = pygame.font.SysFont("Segoe UI Bold", 20)
        font_large = pygame.font.SysFont("Segoe UI Bold", 28)
        font_score = pygame.font.SysFont("Consolas", 40, bold=True)
    except:
        # Fallback for systems without Segoe UI
        font_main = pygame.font.SysFont("Arial", 18)
        font_bold = pygame.font.SysFont("Arial", 20, bold=True)
        font_large = pygame.font.SysFont("Arial", 28, bold=True)
        font_score = pygame.font.SysFont("Courier New", 40, bold=True)

    # Load Model
    model = SnakeNet(board_size=board_size)
    try:
        _load_model_state_compat(model, model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print("Model file not found!")
        return
    except RuntimeError as e:
        print(f"Model error: {e}")
        return

    model.eval()
    
    mcts = None
    if use_mcts:
        mcts = MCTS(model, n_simulations=simulations)
        mcts.reset()

    # Log Configuration
    print("\n" + "="*40)
    print("🐍 SNAKE AI VISUALIZATION CONFIG")
    print("="*40)
    print(f"Model Path:     {model_path}")
    print(f"Board Size:     {board_size}x{board_size}")
    print(f"Method:         {'MCTS' if use_mcts else 'Neural Network Only'}")
    if use_mcts:
        print(f"Default Sims:   {simulations}")
        print(f"Sim Schedule:   {'Enabled' if use_sim_schedule else 'Disabled'}")
        if use_sim_schedule:
            print(f"Generation:     {generation}")
            print(f"Dev Mode:       {'Yes' if dev_mode else 'No'}")
    print(f"Game Speed:     {speed}s per frame")
    print("="*40 + "\n")

    game = SnakeGame(board_size=board_size)
    running = True
    start_time = time.time()
    game_num = 1
    
    guaranteed_win_path = None
    foresight_steps = 0
    total_sims_made = 0

    while running:
        frame_start = time.perf_counter()
        current_time = time.time() - start_time
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_q: running = False

        # AI Prediction
        state_grid = game.get_state()
        pov_numpy = get_pov_state(game, state_grid)
        
        if use_mcts:
            if use_sim_schedule:
                mcts.n_simulations = get_mcts_simulations(
                    generation, len(game.snake), game.board_size, dev_mode=dev_mode
                )
            
            if guaranteed_win_path:
                rel_action = guaranteed_win_path.pop(0)
                p_probs = np.zeros(3)
                p_probs[rel_action] = 1.0
                value = 1.0
                abs_action = (game.direction + (rel_action - 1)) % 4
            else:
                total_sims_made += mcts.n_simulations
                p_probs, entropy, win_path, _sims_done, _timing = mcts.search(game)
                if win_path is not None:
                    sims_to_solve = total_sims_made
                    guaranteed_win_path = win_path
                    foresight_steps = len(guaranteed_win_path)
                    rel_action = guaranteed_win_path.pop(0)
                    p_probs = np.zeros(3)
                    p_probs[rel_action] = 1.0
                    value = 1.0
                    abs_action = (game.direction + (rel_action - 1)) % 4
                else:
                    value = mcts.root.Q if mcts.root else 0.0
                    rel_action = np.argmax(p_probs)
                    abs_action = (game.direction + (rel_action - 1)) % 4
        else:
            input_tensor = torch.tensor(pov_numpy).unsqueeze(0)
            with torch.no_grad():
                p_logits, v = model(input_tensor)
                p_probs = torch.exp(p_logits).cpu().numpy()[0]
                value = v.item()
                rel_action = np.argmax(p_probs)
                abs_action = (game.direction + (rel_action - 1)) % 4

        # --- RENDER ---
        screen.fill(BG_DARK)
        
        # 1. Draw Game Board
        board_rect = pygame.Rect(0, 0, GAME_SIZE, GAME_SIZE)
        pygame.draw.rect(screen, BOARD_BG, board_rect)
        
        # Grid lines
        for i in range(board_size + 1):
            # Vertical
            pygame.draw.line(screen, GRID_COLOR, (i*CELL_SIZE, 0), (i*CELL_SIZE, GAME_SIZE), 1)
            # Horizontal
            pygame.draw.line(screen, GRID_COLOR, (0, i*CELL_SIZE), (GAME_SIZE, i*CELL_SIZE), 1)
        
        # Food with pulse
        pulse = 1.0 + 0.1 * np.sin(current_time * 8)
        fx, fy = game.food
        draw_food(screen, fx, fy, CELL_SIZE, FOOD_COLOR, pulse)
        
        # Snake
        for i, (bx, by) in enumerate(game.snake):
            is_head = (i == 0)
            color = SNAKE_HEAD_COLOR if is_head else SNAKE_BODY_COLOR
            # Lighter version for body gradient-ish look
            if not is_head:
                # Fade color slightly along the body
                factor = 1.0 - (i / len(game.snake)) * 0.4
                color = (int(color[0]*factor), int(color[1]*factor), int(color[2]*factor))
            
            draw_snake_segment(screen, bx, by, CELL_SIZE, color, is_head=is_head, direction=game.direction)

        # 2. Info Panel
        panel_x = GAME_SIZE + 20
        y_offset = 20
        
        # Title & Mode
        if guaranteed_win_path is not None:
            mode_text = "MCTS AUTOPLAYING WIN"
            mode_color = (255, 200, 50)
        elif use_mcts:
            mode_text = "MCTS ENHANCED"
            mode_color = ACCENT_COLOR
        else:
            mode_text = "NEURAL PILOT"
            mode_color = ACCENT_COLOR
            
        mode_label = font_bold.render(mode_text, True, mode_color)
        screen.blit(mode_label, (panel_x, y_offset))
        y_offset += 30
        
        title_label = font_large.render("Snake Intelligence", True, TEXT_COLOR)
        screen.blit(title_label, (panel_x, y_offset))
        y_offset += 50
        
        # Score Display
        score_bg = pygame.Rect(panel_x - 10, y_offset, INFO_WIDTH - 20, 80)
        pygame.draw.rect(screen, (30, 30, 45), score_bg, border_radius=10)
        
        score_label = font_main.render("CURRENT SCORE", True, (150, 150, 170))
        screen.blit(score_label, (panel_x + 10, y_offset + 10))
        
        score_num = font_score.render(f"{game.score:03d}", True, SNAKE_HEAD_COLOR)
        screen.blit(score_num, (panel_x + 10, y_offset + 30))
        y_offset += 100
        
        # Value Estimate
        v_label = font_main.render("State Value Confidence:", True, TEXT_COLOR)
        screen.blit(v_label, (panel_x, y_offset))
        y_offset += 25
        
        # Value bar
        val_bar_w = 250
        pygame.draw.rect(screen, (40, 40, 55), (panel_x, y_offset, val_bar_w, 10), border_radius=5)
        # Map value from [-1, 1] to [0, 1]
        v_normalized = (value + 1) / 2
        v_fill = max(0, min(1, v_normalized))
        v_color = (100, 255, 100) if value > 0 else (255, 100, 100)
        pygame.draw.rect(screen, v_color, (panel_x, y_offset, int(val_bar_w * v_fill), 10), border_radius=5)
        y_offset += 25
        
        # POV Section
        pov_title = font_bold.render("Sensing (POV View)", True, TEXT_COLOR)
        screen.blit(pov_title, (panel_x, y_offset))
        y_offset += 30
        
        POV_SIZE = 12
        draw_pov(screen, pov_numpy, panel_x, y_offset, POV_SIZE)
        y_offset += board_size * POV_SIZE + 30
        
        # Probabilities
        prob_title = font_bold.render("Decision Matrix", True, TEXT_COLOR)
        screen.blit(prob_title, (panel_x, y_offset))
        y_offset += 30
        
        actions = ["Turn Left", "Go Straight", "Turn Right"]
        for i, prob in enumerate(p_probs):
            # Action text
            txt_color = SNAKE_HEAD_COLOR if i == rel_action else (150, 150, 170)
            act_label = font_main.render(actions[i], True, txt_color)
            screen.blit(act_label, (panel_x, y_offset))
            
            # Action bar
            bar_y = y_offset + 22
            pygame.draw.rect(screen, (40, 40, 55), (panel_x, bar_y, 250, 6), border_radius=3)
            bar_color = SNAKE_HEAD_COLOR if i == rel_action else (100, 100, 120)
            pygame.draw.rect(screen, bar_color, (panel_x, bar_y, int(250 * prob), 6), border_radius=3)
            
            # Prob percentage
            perc_label = font_main.render(f"{prob*100:2.0f}%", True, txt_color)
            screen.blit(perc_label, (panel_x + 260, y_offset))
            
            y_offset += 45

        pygame.display.flip()
        
        # Step
        _, reward, done = game.step(abs_action)
        if use_mcts and not guaranteed_win_path: 
            mcts.update_root(rel_action)
        
        if done:
            if game.death_reason == "won" and foresight_steps > 0:
                foresight_pct_game = (foresight_steps / max(1, game.steps)) * 100
                divisor = sims_to_solve if (use_mcts and 'sims_to_solve' in locals() and sims_to_solve > 0) else 1
                foresight_pct_sims = (foresight_steps / divisor) * 100
                print(f"Game {game_num:03d} | Score: {game.score:2d} | Reason: {game.death_reason} | Foresight: {foresight_steps} steps | Foresight % game: {foresight_pct_game:.0f}% | Foresight % sims: {foresight_pct_sims:.1f}%")
            else:
                print(f"Game {game_num:03d} | Score: {game.score:2d} | Reason: {game.death_reason}")

            guaranteed_win_path = None
            foresight_steps = 0
            total_sims_made = 0
            game_num += 1

            # Game Over Overlay
            overlay = pygame.Surface((GAME_SIZE, GAME_SIZE), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))
            
            go_label = font_large.render("GAME OVER", True, (255, 100, 100))
            reason_label = font_main.render(f"Reason: {game.death_reason}", True, (200, 200, 200))
            
            screen.blit(go_label, (GAME_SIZE//2 - go_label.get_width()//2, GAME_SIZE//2 - 20))
            screen.blit(reason_label, (GAME_SIZE//2 - reason_label.get_width()//2, GAME_SIZE//2 + 20))
            pygame.display.flip()
            
            time.sleep(1.0)
            game.reset()
            if use_mcts: mcts.reset()
            
        elapsed = time.perf_counter() - frame_start
        delay = speed - elapsed
        if delay > 0:
            time.sleep(delay)

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
    parser.add_argument("--gen", type=int, default=0, help="Generation index for the MCTS sim schedule (affects sims ramp)")
    parser.add_argument("--no-sim-schedule", action="store_true", help="Disable per-move sim scheduling (use fixed --sims)")
    parser.add_argument("--dev", action="store_true", help="Use dev-mode sim schedule (lower sims; no endgame boost)")
    
    args = parser.parse_args()
    
    visualize(
        model_path=args.model,
        board_size=args.board_size,
        speed=args.speed,
        debug_inputs=args.debug_inputs,
        use_mcts=args.mcts,
        simulations=args.sims,
        generation=int(args.gen),
        use_sim_schedule=not bool(args.no_sim_schedule),
        dev_mode=bool(args.dev),
    )
