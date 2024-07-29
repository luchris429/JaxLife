import jax
import jax.numpy as jnp
import wandb
from world.world import World
from visualizer import Visualizer
import argparse
from datetime import datetime
import copy
import os
import numpy as np
import jax
import jax.numpy as jnp
import wandb
from world.structs import WorldState
from world.world import World
from visualizer import Visualizer
import argparse
from datetime import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--num-agents", type=int, default=128)
parser.add_argument("--num-bots", type=int, default=32)
parser.add_argument("--cell-size", type=int, default=8)
parser.add_argument("--map-size", type=int, default=128)
parser.add_argument("--terrain-alpha", type=float, default=0.005)
parser.add_argument("--max-terrain-energy", type=int, default=8)
parser.add_argument("--hsize", type=int, default=64)
parser.add_argument("--brain", type=str, default="lstm_brain")
parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--terrain-init", type=str, default="perlin")
parser.add_argument("--eat-rate", type=float, default=0.2)
parser.add_argument("--group", type=str, default="unlabeled")
parser.add_argument("--terrain-mod-cost", type=float, default=0.001)
parser.add_argument("--mut-std", type=float, default=0.01)
parser.add_argument("--sweep-id", type=str, default="")
parser.add_argument("--max-terrain-move-cost", type=float, default=4)
parser.add_argument("--max-terrain-push-cost", type=float, default=4)
parser.add_argument("--max-terrain-energy-gain", type=float, default=1)
parser.add_argument("--max-terrain-message-cost", type=float, default=0.25)
parser.add_argument("--max-terrain-reproduce-cost", type=float, default=1)
parser.add_argument("--weather-change-speed", type=float, default=0.001)
parser.add_argument("--perlin_res", type=int, default=4)
parser.add_argument("--hit-cost", type=float, default=0.4)
parser.add_argument("--hit-steal-fraction", type=float, default=0.0)
parser.add_argument("--hit-strength", type=float, default=1.0)
parser.add_argument("--max-reproduce-cost", type=float, default=32.0)
parser.add_argument("--gui", action="store_true")


def gui_loop(world_state: WorldState, visualizer: Visualizer, world: World, config):
    import pygame

    pygame.init()
    upscale = 2
    clock = pygame.time.Clock()
    screen_surface = pygame.display.set_mode(
        (config["IMG_SIZE_W"] * upscale, config["IMG_SIZE_H"] * upscale), display=0
    )

    def _render(pixels):
        screen_surface.fill((0, 0, 0))
        surface = pygame.surfarray.make_surface(np.array(pixels))
        screen_surface.blit(surface, (0, 0))

    step = jax.jit(world.step)
    render = jax.jit(visualizer.render_frame)
    for i in range(config["NUM_WORLD_STEPS"] * config["NUM_OUTER_STEPS"]):
        pygame_events = list(pygame.event.get())
        for event in pygame_events:
            if event.type == pygame.QUIT:
                return
        world_state, metrics = step(world_state, None)
        del metrics["world_state"]
        frame = render(world_state)
        # tile frame upscale by upscale
        frame = np.repeat(np.repeat(frame, upscale, axis=0), upscale, axis=1)
        _render(frame * 255.0)
        pygame.display.flip()
        clock.tick(60)


def main(config):
    world = World(config)
    now = str(datetime.now()).replace(" ", "_")
    render_dir = f"/tmp/renders/{now}"

    if args.render:
        os.makedirs(render_dir, exist_ok=True)
    visualizer = Visualizer(config)

    rng = jax.random.PRNGKey(config["INIT_RNG"])
    world_state = world.initialize(rng)

    if config["GUI"]:
        gui_loop(world_state, visualizer, world, config)
        return

    def world_scan(world_state):
        return jax.lax.scan(world.step, world_state, None, config["NUM_WORLD_STEPS"])

    for i in range(config["NUM_OUTER_STEPS"]):
        out = jax.jit(world_scan)(world_state)

        if args.render and i % config["RENDER_FREQ"] == 0:
            fname = f"{render_dir}/{i:05d}.mp4"
            print(f"RENDERING {fname}")
            visualizer.render(out[1]["world_state"], fname=fname, show=False)
            print(f"SAVING {fname}")
            jnp.save(fname.replace(".mp4", ".npy"), world_state)
            if args.wandb:
                wandb.log({"video": wandb.Video(fname, fps=8, format="mp4")})

        world_state = out[0]
        metrics = {k: wandb.Histogram(v) for k, v in out[1].items() if not k.startswith("world_state")}
        metrics_mean = {f"{k}_mean": v.mean() for k, v in out[1].items() if not k.startswith("world_state")}
        metrics.update(metrics_mean)
        metrics["num_agents_mode"] = jnp.argmax(jnp.bincount(out[1]["num_agents"].astype(jnp.int32)))
        saliency = jax.jit(world.calc_saliency)(world_state)
        metrics.update(saliency)
        metrics["main/comm_saliency"] = saliency["agent_comm_2_saliency"]
        if args.wandb:
            wandb.log(metrics)
        print(i)
        print(out[1]["num_agents"].mean())

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    config = {
        "NUM_WORLD_STEPS": 512,
        "NUM_AGENTS": args.num_agents,
        "MIN_AGENTS": args.num_agents // 4,
        "MAP_SIZE": args.map_size,
        "AGENT_NUM_VIEW_AGENTS": args.num_agents // 8,
        "AGENT_NUM_VIEW_BOTS": 2,
        "CELL_SIZE": args.cell_size,
        "TERRAIN_VIEW_RANGE": 4,
        "ACTION_REP_SIZE": 11,
        "FLATTEN_GRID_OBS": True,
        "INITIAL_ENERGY": 32.0,
        "LIFE_COST": 0.25,
        "AGE_COST": 256.0,
        "MOVE_COST": 0.5,
        "MOVE_SPEED": 0.5,
        "PUSH_COST": 0.25,
        "PUSH_SPEED": 1.0,
        "EAT_COST": 0.1,
        "AGE_SPEED": 1.0,
        "MESSAGE_COST": 0.0,
        "MAX_REPRODUCE_COST": args.max_reproduce_cost,
        "REPRODUCE_FACTOR": 2,  # How many times the reproduce energy cost do you need to have to be able to reproduce.
        "HIT_STEAL_FRACTION": args.hit_steal_fraction,
        "HIT_STRENGTH": args.hit_strength,
        "HIT_COST": args.hit_cost,
        "HIT_DISTANCE_DECAY": 0.5,
        "MUTATION_STD": args.mut_std,
        "HSIZE": args.hsize,
        "ID_SIZE": 8,
        "EAT_RATE": args.eat_rate,
        "IMG_SIZE_H": 512,
        "IMG_SIZE_W": 512,
        "FPS": 8,
        "AGENT_RADIUS": 4,
        "BOT_RADIUS": 8,
        "ACT_TERRAIN_ENERGY_GAIN": 0.1,
        "ACT_TERRAIN_MOVE_COST": 0.1,
        "ACT_TERRAIN_PUSH_COST": 0.1,
        "ACT_TERRAIN_MESSSAGE_COST": 0.1,
        "ACT_TERRAIN_REPRODUCE_COST": 0.1,
        "MAX_TERRAIN_ENERGY": args.max_terrain_energy,
        "MAX_TERRAIN_MOVE_COST": args.max_terrain_move_cost,
        "MAX_TERRAIN_PUSH_COST": args.max_terrain_push_cost,
        "MAX_TERRAIN_ENERGY_GAIN": args.max_terrain_energy_gain,
        "MAX_TERRAIN_MESSAGE_COST": args.max_terrain_message_cost,
        "MAX_TERRAIN_REPRODUCE_COST": args.max_terrain_reproduce_cost,
        "PERLIN_RES": args.perlin_res,
        "WEATHER_CHANGE_SPEED": args.weather_change_speed,
        "TERRAIN_ALPHA": args.terrain_alpha,
        "TERRAIN_GAIN_SCALING": 1.5,
        "RETURN_WORLD_STATE": False,
        "ONLY_RESET_WHEN_NO_AGENTS": True,
        "BRAIN": args.brain,
        "TERRAIN_INIT": args.terrain_init,
        "TERRAIN_MOD_COST": args.terrain_mod_cost,
        "TERRAIN_PERIOD": 182.5,
        "INIT_RNG": args.seed,
        "NUM_BOTS": args.num_bots,
        "BOT_NUM_VIEW_AGENTS": 4,
        "BOT_NUM_VIEW_BOTS": 4,
        "BOT_BRAIN": "complex_program_brain",
        "BOT_INITIAL_ENERGY": 64.0,
        "MESSAGE_SIZE": 16,
        "BOT_MEMORY_SIZE": 16,
        "BOT_PROGRAM_SIZE": 16,
        "BOT_STEP_RATIO": 4,
        "IGNORE_AGENT_UPDATE": False,
        "RENDER_FREQ": 8,
        "RETURN_WORLD_STATE": True,
        "GUI": args.gui,
        "NUM_OUTER_STEPS": 128,
    }
    now = str(datetime.now()).replace(" ", "_")
    config["NOW"] = now

    if args.wandb and not args.tune:
        run = wandb.init(
            project="JaxLife",
            config=config,
            group=args.group,
        )
    main(config)
