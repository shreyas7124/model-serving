#!/usr/bin/env python3
"""
Generate docker-compose.vllm.generated.yml from placement env.

Maps MODEL_DEPLOY_MODE + VLLM_REPLICA_COUNT + TENSOR_PARALLEL_SIZE to N vLLM
services and BACKEND_URLS for HF apps.

Usage (from repo root):
  export MODEL_DEPLOY_MODE=hybrid
  export VLLM_REPLICA_COUNT=2
  export TENSOR_PARALLEL_SIZE=2
  export VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
  export GPU_DEVICES=0,1,2,3
  python deploy/scripts/generate_vllm_compose.py
  docker compose -f deploy/docker-compose.vllm.generated.yml up -d
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def split_csv(raw: str):
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-vLLM compose file")
    parser.add_argument(
        "-o",
        "--output",
        default="deploy/docker-compose.vllm.generated.yml",
        help="Output compose path",
    )
    parser.add_argument(
        "--include-apps",
        action="store_true",
        help="Also emit hf-ide-assistant service wired to BACKEND_URLS",
    )
    args = parser.parse_args()

    mode = os.getenv("MODEL_DEPLOY_MODE", os.getenv("LLM_DEPLOY_MODE", "hybrid")).strip().lower()
    if mode not in ("replica", "sharded", "hybrid"):
        mode = "hybrid"

    tp = max(1, env_int("TENSOR_PARALLEL_SIZE", env_int("TP_SIZE", 1)))
    pp = max(1, env_int("PIPELINE_PARALLEL_SIZE", env_int("PP_SIZE", 1)))
    gpus_per = max(tp * pp, env_int("GPU_PER_REPLICA", tp * pp), 1)

    if mode == "sharded":
        replicas = 1
    else:
        replicas = max(1, env_int("VLLM_REPLICA_COUNT", env_int("REPLICA_COUNT", 1)))

    model = os.getenv("VLLM_MODEL", os.getenv("HF_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"))
    image = os.getenv("VLLM_IMAGE", "vllm/vllm-openai:latest")
    max_model_len = env_int("VLLM_MAX_MODEL_LEN", 8192)
    port = env_int("VLLM_PORT", 8000)
    gpu_list = split_csv(os.getenv("GPU_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "")))
    needed = replicas * gpus_per
    if gpu_list and len(gpu_list) < needed:
        raise SystemExit(
            f"Not enough GPU_DEVICES ({len(gpu_list)}) for {replicas} replicas x {gpus_per} GPUs "
            f"(need {needed}). Set GPU_DEVICES or lower VLLM_REPLICA_COUNT / TENSOR_PARALLEL_SIZE."
        )
    if not gpu_list:
        # synthetic indices for documentation; compose still requests device reservations
        gpu_list = [str(i) for i in range(needed)]

    enable_prefix = os.getenv("VLLM_ENABLE_PREFIX_CACHING", "true").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    mooncake_kv = os.getenv("VLLM_MOONCAKE_KV", "false").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # Default: enable Mooncake store sidecar note when multi-replica
    if replicas > 1 and os.getenv("VLLM_MOONCAKE_KV", "") == "":
        mooncake_kv = True

    services = {}
    backend_urls = []
    gpu_idx = 0

    for r in range(replicas):
        name = f"vllm-{r}"
        devices = gpu_list[gpu_idx : gpu_idx + gpus_per]
        gpu_idx += gpus_per
        cvd = ",".join(devices)
        host_port = port + r
        cmd = [
            "--model",
            model,
            "--tensor-parallel-size",
            str(tp),
            "--max-model-len",
            str(max_model_len),
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
        if enable_prefix:
            cmd.append("--enable-prefix-caching")
        # pipeline parallel if >1 (vllm flag name varies; document as env for operator)
        if pp > 1:
            cmd.extend(["--pipeline-parallel-size", str(pp)])

        env = {
            "CUDA_VISIBLE_DEVICES": cvd,
            "HUGGING_FACE_HUB_TOKEN": "${HUGGING_FACE_HUB_TOKEN:-}",
        }
        if mooncake_kv:
            env["MOONCAKE_MASTER"] = os.getenv("MOONCAKE_MASTER", "mooncake-master:50051")
            env["VLLM_KV_CACHE_BACKEND_NOTE"] = (
                "Configure vLLM Mooncake KV connector per your vLLM version; "
                "GPU working set remains primary; Mooncake is hierarchical overflow/share."
            )

        services[name] = {
            "image": image,
            "command": cmd,
            "environment": env,
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [
                            {
                                "driver": "nvidia",
                                "device_ids": devices,
                                "capabilities": ["gpu"],
                            }
                        ]
                    }
                }
            },
            "ports": [f"{host_port}:8000"],
            "volumes": [
                "${HF_CACHE:-~/.cache/huggingface}:/root/.cache/huggingface",
            ],
            "ipc": "host",
        }
        backend_urls.append(f"http://{name}:8000/v1/chat/completions")

    if mooncake_kv:
        services["mooncake-master"] = {
            "image": os.getenv("MOONCAKE_IMAGE", "mooncake-store:latest"),
            "profiles": ["mooncake"],
            "ports": ["50051:50051"],
            "environment": {
                "NOTE": (
                    "Replace image with your Mooncake store build; "
                    "see https://kvcache-ai.github.io/Mooncake/"
                ),
            },
        }

    backend_csv = ",".join(backend_urls)

    if args.include_apps:
        services["hf-ide-assistant"] = {
            "build": {"context": "..", "dockerfile": "deploy/Dockerfile.python"},
            "working_dir": "/app/hf-ide-assistant",
            "command": ["python", "server.py"],
            "environment": {
                "LOAD_MODEL_WEIGHTS": "false",
                "MODEL_DEPLOY_MODE": mode,
                "BACKEND_URLS": backend_csv,
                "BACKEND_STRATEGY": os.getenv("BACKEND_STRATEGY", "round_robin"),
                "HF_MODEL_NAME": model,
                "HOST": "0.0.0.0",
                "PORT": "8081",
                "API_KEY": os.getenv("API_KEY", "hf-coding-assistant-key"),
                "TENSOR_PARALLEL_SIZE": str(tp),
            },
            "ports": ["8081:8081"],
            "depends_on": [f"vllm-{i}" for i in range(replicas)],
            "volumes": ["..:/app"],
        }

    # Emit YAML manually (avoid PyYAML dependency)
    lines = [
        f"# AUTO-GENERATED by deploy/scripts/generate_vllm_compose.py",
        f"# MODE={mode} replicas={replicas} tp={tp} pp={pp} gpus_per_replica={gpus_per}",
        f"# BACKEND_URLS={backend_csv}",
        f"# Model={model}",
        "# GPU working-set KV stays in vLLM; optional Mooncake is hierarchical overflow/share.",
        "services:",
    ]

    def emit_service(name: str, svc: dict, indent: int = 2) -> None:
        sp = " " * indent
        lines.append(f"{sp}{name}:")
        if "image" in svc:
            lines.append(f"{sp}  image: {svc['image']}")
        if "build" in svc:
            b = svc["build"]
            lines.append(f"{sp}  build:")
            lines.append(f"{sp}    context: {b['context']}")
            lines.append(f"{sp}    dockerfile: {b['dockerfile']}")
        if "working_dir" in svc:
            lines.append(f"{sp}  working_dir: {svc['working_dir']}")
        if "command" in svc:
            lines.append(f"{sp}  command:")
            for c in svc["command"]:
                lines.append(f"{sp}    - {c!r}".replace("\'", '"') if False else f"{sp}    - \"{c}\"")
        if "environment" in svc:
            lines.append(f"{sp}  environment:")
            for k, v in svc["environment"].items():
                # quote values with special chars
                vs = str(v).replace('"', '\\"')
                lines.append(f"{sp}    {k}: \"{vs}\"")
        if "ports" in svc:
            lines.append(f"{sp}  ports:")
            for p in svc["ports"]:
                lines.append(f"{sp}    - \"{p}\"")
        if "volumes" in svc:
            lines.append(f"{sp}  volumes:")
            for v in svc["volumes"]:
                lines.append(f"{sp}    - {v}")
        if "ipc" in svc:
            lines.append(f"{sp}  ipc: {svc['ipc']}")
        if "profiles" in svc:
            lines.append(f"{sp}  profiles:")
            for p in svc["profiles"]:
                lines.append(f"{sp}    - {p}")
        if "depends_on" in svc:
            lines.append(f"{sp}  depends_on:")
            for d in svc["depends_on"]:
                lines.append(f"{sp}    - {d}")
        if "deploy" in svc:
            lines.append(f"{sp}  deploy:")
            lines.append(f"{sp}    resources:")
            lines.append(f"{sp}      reservations:")
            lines.append(f"{sp}        devices:")
            dev = svc["deploy"]["resources"]["reservations"]["devices"][0]
            lines.append(f"{sp}          - driver: {dev['driver']}")
            lines.append(f"{sp}            capabilities: [gpu]")
            ids = ", ".join(f'\"{x}\"' for x in dev["device_ids"])
            lines.append(f"{sp}            device_ids: [{ids}]")

    for name, svc in services.items():
        emit_service(name, svc)

    out = Path(args.output)
    if not out.is_absolute():
        # relative to repo root (cwd)
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")

    # Also write env snippet for apps
    env_out = out.with_suffix(".env")
    env_out.write_text(
        "\n".join(
            [
                f"MODEL_DEPLOY_MODE={mode}",
                f"VLLM_REPLICA_COUNT={replicas}",
                f"TENSOR_PARALLEL_SIZE={tp}",
                f"PIPELINE_PARALLEL_SIZE={pp}",
                f"GPU_PER_REPLICA={gpus_per}",
                f"VLLM_MODEL={model}",
                f"HF_MODEL_NAME={model}",
                f"BACKEND_URLS={backend_csv}",
                f"LOAD_MODEL_WEIGHTS=false",
                f"BACKEND_STRATEGY={os.getenv('BACKEND_STRATEGY', 'round_robin')}",
            ]
        )
        + "\n"
    )
    print(f"Wrote {out}")
    print(f"Wrote {env_out}")
    print(f"BACKEND_URLS={backend_csv}")


if __name__ == "__main__":
    main()
