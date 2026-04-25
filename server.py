"""OpenEnv-compliant environment server for the Constrained Refactor Gauntlet."""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os

from environment.episode_generator import EpisodeGenerator
from environment.track_a import CodeQualityEvaluator
from environment.track_b import ComplianceChecker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPISODE_TTL_SECONDS = 1800  # 30 minutes
CLEANUP_INTERVAL_SECONDS = 300  # 5 minutes
MAX_CONTENT_LENGTH = 1_048_576  # 1 MB

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Constrained Refactor Gauntlet",
    description="RL environment where an agent refactors legacy Python under 150 cascading engineering rules.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    curriculum_level: Optional[int] = None
    seed: Optional[int] = None

class ActionRequest(BaseModel):
    episode_id: str
    action: Dict[str, Any]

# ---------------------------------------------------------------------------
# Episode Context
# ---------------------------------------------------------------------------

class EpisodeContext:
    def __init__(self, base_dir: str):
        self.generator = EpisodeGenerator(base_dir)
        self.track_a = CodeQualityEvaluator()
        self.track_b = ComplianceChecker(
            os.path.join(os.path.dirname(base_dir), "ENGINEERING_STANDARDS.md")
        )
        self.files: Dict[str, str] = {}
        self.steps_remaining: int = 70
        self.created_at: float = time.time()

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
active_episodes: Dict[str, EpisodeContext] = {}
BASE_DIR = os.path.join(os.path.dirname(__file__), "environment/base_codebase")
START_TIME = time.time()

# ---------------------------------------------------------------------------
# Startup: episode cleanup task
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def setup_cleanup():
    """Periodically remove episodes that have exceeded the TTL."""
    async def cleanup_loop():
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            cutoff = time.time() - EPISODE_TTL_SECONDS
            expired = [eid for eid, ctx in active_episodes.items() if ctx.created_at < cutoff]
            for eid in expired:
                del active_episodes[eid]
            if expired:
                logger.info("Cleaned up %d expired episodes", len(expired))

    asyncio.create_task(cleanup_loop())

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset_env(req: Optional[ResetRequest] = None):
    ctx = EpisodeContext(BASE_DIR)
    if req and req.curriculum_level is not None:
        ctx.generator.curriculum.level = req.curriculum_level

    episode_data = ctx.generator.generate()
    episode_id = episode_data["episode_id"]
    ctx.files = episode_data["files"]

    # Derive a deterministic seed from the episode ID when not provided
    seed = (req.seed if req and req.seed is not None
            else hash(episode_id) % (2 ** 31))

    ctx.track_a.evaluate(ctx.files)
    report = ctx.track_b.reset(ctx.files, episode_data["rules_active"], seed=seed)

    active_episodes[episode_id] = ctx
    logger.info("Created episode %s (curriculum=%d, rules=%d)",
                episode_id, episode_data["curriculum_level"], len(episode_data["rules_active"]))

    observation = {
        "files": ctx.files,
        "active_rules_count": len(episode_data["rules_active"]),
        "steps_remaining": ctx.steps_remaining,
        "violation_report": {
            "newly_triggered": list(report.newly_triggered),
            "newly_resolved": list(report.newly_resolved),
            "still_outstanding": list(report.still_outstanding),
            "conflict_flags": report.conflict_flags,
        },
        "curriculum_level": episode_data["curriculum_level"],
    }

    return {
        "episode_id": episode_id,
        "observation": observation,
        "info": {},
    }


@app.post("/step")
async def step_env(req: ActionRequest):
    episode_id = req.episode_id
    if episode_id not in active_episodes:
        raise HTTPException(status_code=404, detail="Episode not found")

    ctx = active_episodes[episode_id]
    action = req.action
    tool = action.get("tool")
    args = action.get("args", {})

    diff = ""
    if tool == "edit_file":
        filename = args.get("filename")
        content = args.get("content")
        if filename and content:
            # Enforce content size limit
            if len(content) > MAX_CONTENT_LENGTH:
                raise HTTPException(status_code=400, detail="Content exceeds 1 MB limit")
            ctx.files[filename] = content
            diff = f"Edited {filename}"
            ctx.track_a.evaluate(ctx.files)
        else:
            diff = "edit_file with missing args"

    elif tool == "run_tests":
        ctx.track_a.evaluate(ctx.files)
        diff = "Ran tests"

    elif tool == "check_compliance":
        diff = "Checked compliance"

    elif tool == "read_file":
        diff = "Read file"

    else:
        diff = "Unknown/Finish"

    # Always compute the compliance report (fixes unbound `report` bug)
    report = ctx.track_b.step(action, diff)

    ctx.steps_remaining -= 1
    done = ctx.steps_remaining <= 0 or tool == "finish"

    # Build the observation — for read_file, include only the requested file
    obs_files = ctx.files
    if tool == "read_file":
        requested = args.get("filename")
        if requested and requested in ctx.files:
            obs_files = {requested: ctx.files[requested]}

    observation = {
        "files": obs_files,
        "steps_remaining": ctx.steps_remaining,
        "violation_report": {
            "newly_triggered": list(report.newly_triggered),
            "newly_resolved": list(report.newly_resolved),
            "still_outstanding": list(report.still_outstanding),
            "conflict_flags": report.conflict_flags,
        },
    }

    reward = None
    info: Dict[str, Any] = {}
    if done:
        code_score = ctx.track_a.evaluate(ctx.files)
        compliance_score = ctx.track_b.get_score()
        reward = code_score.total * compliance_score

        info = {
            "final_reward": reward,
            "code_score": {
                "test_pass_rate": code_score.test_pass_rate,
                "lint_improvement": code_score.lint_improvement,
                "complexity_reduction": code_score.complexity_reduction,
                "module_size_compliance": code_score.module_size_compliance,
                "total": code_score.total,
            },
            "compliance_score": compliance_score,
        }

        del active_episodes[episode_id]
        logger.info("Episode %s completed — reward=%.3f", episode_id, reward)

    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "active_episodes": len(active_episodes),
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }
