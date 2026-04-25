#!/usr/bin/env python
"""Roll out a trained policy in a MotLab env, optionally recording video.

    python scripts/play.py --env cartpole --policy runs/cartpole/.../model.pt
    python scripts/play.py --env cartpole --policy ... --video out.mp4
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time

from absl import app, flags

logger = logging.getLogger(__name__)

_ENV = flags.DEFINE_string("env", "cartpole", "Environment name.")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 16, "Number of envs to roll out.")
_POLICY = flags.DEFINE_string("policy", None, "Path to saved policy (.pt).")
_STEPS = flags.DEFINE_integer("steps", 1000, "Number of control steps.")
_VIDEO = flags.DEFINE_string("video", None, "If set, write rendered mp4 to this path.")
_FPS = flags.DEFINE_integer("fps", 50, "Video frame rate.")
_CAMERA = flags.DEFINE_integer("camera", 0, "Camera index in the MJCF.")
_WIDTH = flags.DEFINE_integer("width", 640, "Render frame width (px).")
_HEIGHT = flags.DEFINE_integer("height", 480, "Render frame height (px).")


def _take_image_blocking(task, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        img = task.take_image()
        if img is not None:
            return img
        time.sleep(0.005)
    raise RuntimeError(f"Render capture timed out (state={task.state})")


def _record(inner_env, step_fn, get_obs_fn, num_steps, out_path, fps, camera_idx, width, height):
    """Run ``step_fn``/``get_obs_fn`` for ``num_steps`` and dump an mp4."""
    import motrixsim.render as render

    ffmpeg_bin = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found on PATH; required for mp4 encoding")

    engine = inner_env.scene.articulations["robot"].engine
    # Switch the model camera to "image" target before launching the render app —
    # required for headless capture (the default "window" target needs a display).
    engine.model.cameras[camera_idx].set_render_target("image", w=width, h=height)
    app_ = render.RenderApp(headless=True)
    app_.launch(engine.model, batch=engine.num_envs)
    cam = app_.get_camera(camera_idx)
    cam.active = True

    tmpdir = tempfile.mkdtemp(prefix="motlab_frames_")
    logger.info("Writing frames to %s", tmpdir)
    obs = get_obs_fn()
    for i in range(num_steps):
        obs = step_fn(obs)
        # Submit capture first, then sync(wait=True) drives the renderer and
        # blocks until the capture completes.
        task = cam.capture()
        app_.sync(engine.data, wait=True)
        img = _take_image_blocking(task)
        img.save_to_disk(os.path.join(tmpdir, f"frame_{i:05d}.png"))
    n_written = len([f for f in os.listdir(tmpdir) if f.endswith(".png")])
    logger.info("Rendered %d frames (wrote %d files); encoding to %s", num_steps, n_written, out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            out_path,
        ],
        check=True,
    )
    logger.info("Saved %s", out_path)
    shutil.rmtree(tmpdir, ignore_errors=True)


def main(_argv) -> None:
    import torch

    import motlab
    import motlab_tasks  # noqa: F401  (registers built-in envs)

    if _VIDEO.value is not None and _NUM_ENVS.value != 1:
        logger.info("--video set: forcing --num-envs=1 for clean recording")
    num_envs = 1 if _VIDEO.value else _NUM_ENVS.value

    # ---- zero-action path (no policy) -------------------------------------
    if _POLICY.value is None:
        logger.warning("--policy not given; running zero-action rollout.")
        cfg = motlab.make_cfg(_ENV.value)
        cfg.scene.num_envs = num_envs
        env = motlab.ManagerBasedRLEnv(cfg, device="cpu")
        env.reset()
        zeros = torch.zeros(env.num_envs, env.action_dim, dtype=torch.float32, device=env.device)

        def step_fn(_o):
            env.step(zeros)
            return None

        if _VIDEO.value:
            _record(env, step_fn, lambda: None, _STEPS.value, _VIDEO.value, _FPS.value, _CAMERA.value, _WIDTH.value, _HEIGHT.value)
            return
        for _ in range(_STEPS.value):
            step_fn(None)
        return

    # ---- policy path ------------------------------------------------------
    from motlab_rl.rslrl.torch.train import RslrlTrainer

    trainer = RslrlTrainer(env_name=_ENV.value, cfg_override={"num_envs": num_envs})
    trainer.runner.load(_POLICY.value, load_optimizer=False)
    policy = trainer.runner.get_inference_policy(device=trainer.runner.device)

    @torch.no_grad()
    def step_fn(obs):
        actions = policy(obs)
        new_obs, _, _, _ = trainer.env.step(actions)
        return new_obs

    if _VIDEO.value:
        _record(
            trainer.env.env,
            step_fn,
            trainer.env.get_observations,
            _STEPS.value, _VIDEO.value, _FPS.value, _CAMERA.value, _WIDTH.value, _HEIGHT.value,
        )
        return

    obs = trainer.env.get_observations()
    for _ in range(_STEPS.value):
        obs = step_fn(obs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app.run(main)
