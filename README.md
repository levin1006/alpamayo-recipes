# Alpamayo Recipes

A collection of end-to-end Alpamayo recipes for multiple versions (v1, v1.5, and beyond), designed
to help developers quickly build, adapt, and deploy Alpamayo-based applications. This repo
brings together battle-tested workflows across the Alpamayo ecosystem, including post-training
recipes (supervised fine-tuning and reinforcement learning), quantization recipes, etc.
Whether you are experimenting locally or building a full production stack, this repository is
intended as the primary starting point for developers to learn, customize, and extend
Alpamayo for their own use cases.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the repository layout, recipe packaging conventions,
and guidance on adding new recipes for released Alpamayo models.

## Recipes

Each recipe folder contains its own README with installation and training instructions.

| Recipe | Description |
|--------|-------------|
| [`recipes/alpamayo1_sft/`](recipes/alpamayo1_sft/README.md) | Alpamayo 1 supervised fine-tuning (HuggingFace Trainer + DeepSpeed) |
| [`recipes/alpamayo1_5_sft/`](recipes/alpamayo1_5_sft/README.md) | Alpamayo 1.5 SFT (HuggingFace Trainer + DeepSpeed) |
| [`recipes/alpamayo1_x_rl/`](recipes/alpamayo1_x_rl/README.md) | Alpamayo 1 and 1.5 RL post-training (Cosmos-RL / GRPO) |

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `scripts/curate_pai_samples.py` | Curate a subset of PAI samples |
| `scripts/convert_checkpoint.py` | Convert between Alpamayo 1 and 1.5 checkpoints |
| `scripts/convert_release_config_to_training.py` | Convert a release checkpoint to training format |
| `scripts/convert_cosmos_rl_checkpoint.py` | Convert a Cosmos-RL checkpoint to HuggingFace format |
| `scripts/download_pai.py` | Download the Physical AI AV dataset from HuggingFace |
| `scripts/check_pai_download_status.py` | Inventory local PAI components and nav-demo chunk coverage |
| `scripts/plan_pai_nav_download.py` | Print or execute the required PAI nav-demo chunk download command |
| `scripts/a15_sft_readme_00_setup_env.sh` | Set up the Alpamayo 1.5 SFT recipe environment in tmux |
| `scripts/a15_sft_readme_01_download_pai_nav_chunks.sh` | Download README PAI nav-demo chunks in tmux |
| `scripts/a15_sft_readme_02_download_checkpoint.sh` | Download `nvidia/Alpamayo-1.5-10B` in tmux |
| `scripts/a15_sft_readme_03_convert_checkpoint_to_a1.sh` | Convert the raw Alpamayo 1.5 checkpoint to A1-format for SFT |
| `scripts/a15_sft_readme_04_verify_a1_checkpoint.sh` | Verify the converted A1-format checkpoint metadata |
| `scripts/a15_sft_readme_05_stage1_nav_smoke.sh` | Run bounded Stage 1 PAI nav SFT smoke in tmux |
| `scripts/a15_sft_readme_06_stage2_nav_smoke.sh` | Run bounded Stage 2 PAI nav SFT smoke in tmux |
| `scripts/a15_sft_readme_07_eval_stage2_nav.sh` | Evaluate the bounded Stage 2 nav checkpoint in tmux |
