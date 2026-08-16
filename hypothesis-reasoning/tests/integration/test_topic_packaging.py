from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pydantic


def test_built_wheel_can_load_all_packaged_prompts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    source_copy = tmp_path / "source"
    shutil.copytree(
        project_root,
        source_copy,
        ignore=shutil.ignore_patterns(
            ".cache",
            ".pytest_cache",
            ".venv",
            "build",
            "*.egg-info",
            "__pycache__",
        ),
    )
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(source_copy),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheel = next(wheel_dir.glob("hypothesis_reasoning-*.whl"))
    dependency_root = Path(pydantic.__file__).resolve().parent.parent
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(wheel), str(dependency_root)))
    smoke = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from hypothesis_reasoning.critique import AdversarialCritic; "
                "from hypothesis_reasoning.generation import PromptRegistry; "
                "from hypothesis_reasoning.models import ReasoningType; "
                "from hypothesis_reasoning.refinement import Refiner; "
                "from hypothesis_reasoning.topic_discovery import TopicDiscoverer; "
                "from hypothesis_reasoning.topic_selection import "
                "OneShotTopicSelector, TopicSelector; "
                "TopicDiscoverer(object()); TopicSelector(object()); "
                "OneShotTopicSelector(object(), output_dir=Path('.')); "
                "AdversarialCritic(object()); Refiner(object()); "
                "registry = PromptRegistry(); "
                "registry.get(ReasoningType.INDUCTIVE); "
                "registry.get(ReasoningType.DEDUCTIVE)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert smoke.returncode == 0, smoke.stderr
