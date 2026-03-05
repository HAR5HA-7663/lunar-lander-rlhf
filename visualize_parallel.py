"""
Parallel LunarLander Visualizer
────────────────────────────────
50 agents running simultaneously in a 10×5 grid.

Controls
  B      → switch to Baseline PPO
  A      → switch to Aligned PPO
  SPACE  → toggle between agents
  Q/ESC  → quit

Auto-saves a GIF whenever any agent lands successfully and beats
the current best return for that agent type.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import gymnasium as gym
from stable_baselines3 import PPO
import imageio
from PIL import Image, ImageDraw

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
BASELINE_CKPT = ROOT / 'checkpoints' / 'baseline_ppo'   # SB3 appends .zip automatically
ALIGNED_CKPT  = ROOT / 'checkpoints' / 'aligned_ppo'
GIF_DIR       = ROOT / 'checkpoints'

# ── Layout ────────────────────────────────────────────────────────────────────
N_COLS   = 10
N_ROWS   = 5
N_ENVS   = N_COLS * N_ROWS      # 50
CELL_W   = 120                  # px per mini-env
CELL_H   = 80
HEADER_H = 52
WIN_W    = N_COLS * CELL_W      # 1200
WIN_H    = N_ROWS * CELL_H + HEADER_H  # 452

FPS_CAP  = 30
GIF_FPS  = 30

# ── Colours ───────────────────────────────────────────────────────────────────
BG       = (12,  12,  22)
HDR_BG   = (18,  18,  35)
WHITE    = (230, 230, 255)
DIM      = (110, 110, 160)
GREEN    = (80,  220, 120)
ORANGE   = (255, 165,  60)


# ── Helpers ───────────────────────────────────────────────────────────────────
def add_watermark(frames, text="@HAR5HA"):
    out = []
    for frame in frames:
        img  = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        w, h = img.size
        x, y = w // 2 - 28, int(h * 0.84)
        draw.text((x + 1, y + 1), text, fill=(80, 80, 80))
        draw.text((x,     y    ), text, fill=(30, 30, 30))
        out.append(np.array(img))
    return out


def record_episode(model, gif_path):
    """Run one clean episode and save as GIF. Returns total return."""
    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    frames, obs, done, total = [], env.reset()[0], False, 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        total += r
        done = terminated or truncated
    env.close()
    frames = add_watermark(frames)
    imageio.mimsave(str(gif_path), frames, fps=GIF_FPS)
    return total


def resize_frame(frame):
    """Scale 600×400 numpy frame down to CELL_W×CELL_H."""
    return np.array(
        Image.fromarray(frame).resize((CELL_W, CELL_H), Image.BILINEAR)
    )


def load_models():
    available = {}
    if BASELINE_CKPT.with_suffix('.zip').exists():
        print(f"  Loading baseline … {BASELINE_CKPT.name}.zip")
        available['Baseline PPO'] = PPO.load(str(BASELINE_CKPT))
    else:
        print(f"  ⚠  Baseline checkpoint not found: {BASELINE_CKPT}.zip")

    if ALIGNED_CKPT.with_suffix('.zip').exists():
        print(f"  Loading aligned  … {ALIGNED_CKPT.name}.zip")
        available['Aligned PPO'] = PPO.load(str(ALIGNED_CKPT))
    else:
        print(f"  ⚠  Aligned checkpoint not found: {ALIGNED_CKPT}.zip")

    if not available:
        sys.exit("No checkpoints found. Run notebooks 01 and 05 first.")
    return available


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("LunarLander Parallel Visualizer")
    print("================================")
    models     = load_models()
    agent_keys = list(models.keys())
    agent_idx  = 0

    # ── Init pygame ──────────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(
        "LunarLander Parallel  |  B=Baseline  A=Aligned  SPACE=toggle  Q=quit"
    )
    font     = pygame.font.SysFont('monospace', 16, bold=True)
    font_sm  = pygame.font.SysFont('monospace', 12)
    clock    = pygame.time.Clock()

    # ── Create 50 envs ───────────────────────────────────────────────────────
    print(f"Creating {N_ENVS} environments …")
    envs = [gym.make('LunarLander-v3', render_mode='rgb_array')
            for _ in range(N_ENVS)]
    obs  = [env.reset()[0] for env in envs]

    # Per-env tracking
    ep_return  = [0.0] * N_ENVS
    ep_count   = [0  ] * N_ENVS

    # Per-agent-type best return (for GIF threshold)
    best_return = {k: -float('inf') for k in agent_keys}
    gif_count   = 0
    gif_msg     = ''        # on-screen notification
    gif_msg_ttl = 0         # frames to show notification

    print(f"Starting visualizer. Agent: {agent_keys[agent_idx]}\n")

    running = True
    while running:

        # ── Events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_b and 'Baseline PPO' in models:
                    agent_idx = agent_keys.index('Baseline PPO')
                    print(f"→ Switched to Baseline PPO")
                elif event.key == pygame.K_a and 'Aligned PPO' in models:
                    agent_idx = agent_keys.index('Aligned PPO')
                    print(f"→ Switched to Aligned PPO")
                elif event.key == pygame.K_SPACE:
                    agent_idx = (agent_idx + 1) % len(agent_keys)
                    print(f"→ Switched to {agent_keys[agent_idx]}")

        model     = models[agent_keys[agent_idx]]
        agent_tag = agent_keys[agent_idx]

        # ── Step all envs ────────────────────────────────────────────────────
        obs_batch        = np.array(obs)
        actions_batch, _ = model.predict(obs_batch, deterministic=True)

        cells = []
        for i, env in enumerate(envs):
            new_obs, reward, terminated, truncated, _ = env.step(int(actions_batch[i]))
            ep_return[i] += reward

            cells.append(resize_frame(env.render()))

            done = terminated or truncated
            if done:
                # Successful landing → maybe save GIF
                if terminated and reward == 100 and ep_return[i] > best_return[agent_tag]:
                    prev_best = best_return[agent_tag]
                    best_return[agent_tag] = ep_return[i]

                    gif_name = f'success_{gif_count:03d}_{agent_tag.replace(" ", "_")}.gif'
                    gif_path = GIF_DIR / gif_name

                    print(f"  ✓ New best {agent_tag}: {ep_return[i]:.1f}  "
                          f"(was {prev_best:.1f})  → saving {gif_name} …", end='', flush=True)
                    saved_return = record_episode(model, gif_path)
                    gif_count += 1
                    print(f" done  (recorded return={saved_return:.1f})")

                    gif_msg     = f"GIF saved: {gif_name}  (return={saved_return:.1f})"
                    gif_msg_ttl = FPS_CAP * 4   # show for 4 seconds

                best_return[agent_tag] = max(best_return[agent_tag], ep_return[i])
                obs[i], _  = env.reset()
                ep_return[i] = 0.0
                ep_count[i] += 1
            else:
                obs[i] = new_obs

        # ── Build grid ───────────────────────────────────────────────────────
        rows = [
            np.concatenate(cells[r * N_COLS:(r + 1) * N_COLS], axis=1)
            for r in range(N_ROWS)
        ]
        grid = np.concatenate(rows, axis=0)   # (N_ROWS*CELL_H, WIN_W, 3)
        surf = pygame.surfarray.make_surface(grid.transpose(1, 0, 2))

        # ── Draw ─────────────────────────────────────────────────────────────
        screen.fill(BG)
        screen.blit(surf, (0, HEADER_H))

        # Header background
        pygame.draw.rect(screen, HDR_BG, (0, 0, WIN_W, HEADER_H))
        pygame.draw.line(screen, (40, 40, 80), (0, HEADER_H - 1), (WIN_W, HEADER_H - 1))

        fps          = clock.get_fps()
        total_eps    = sum(ep_count)
        best_val     = best_return[agent_tag]
        best_str     = f"{best_val:.1f}" if best_val > -1e8 else "---"
        agent_colour = GREEN if 'Aligned' in agent_tag else ORANGE

        # Row 1
        line1 = (
            f"Agent: "
        )
        x = 10
        screen.blit(font.render("Agent: ", True, WHITE), (x, 8))
        x += font.size("Agent: ")[0]
        screen.blit(font.render(agent_tag, True, agent_colour), (x, 8))
        x += font.size(agent_tag)[0] + 30

        screen.blit(font.render(f"Episodes: {total_eps:5d}", True, WHITE), (x, 8))
        x += font.size(f"Episodes: {total_eps:5d}")[0] + 30

        screen.blit(font.render(f"Best: {best_str}", True, WHITE), (x, 8))
        x += font.size(f"Best: {best_str}")[0] + 30

        screen.blit(font.render(f"GIFs: {gif_count}", True, WHITE), (x, 8))
        x += font.size(f"GIFs: {gif_count}")[0] + 30

        screen.blit(font.render(f"{fps:4.0f} fps", True, DIM), (x, 8))

        # Row 2 — hints / GIF notification
        if gif_msg_ttl > 0:
            screen.blit(font_sm.render(f"✓ {gif_msg}", True, GREEN), (10, 32))
            gif_msg_ttl -= 1
        else:
            screen.blit(font_sm.render(
                "B = Baseline   A = Aligned   SPACE = toggle   Q = quit",
                True, DIM), (10, 32))

        pygame.display.flip()
        clock.tick(FPS_CAP)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    for env in envs:
        env.close()
    pygame.quit()
    print(f"\nClosed. {gif_count} GIFs saved to {GIF_DIR}")


if __name__ == '__main__':
    main()
