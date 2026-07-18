PYTHON ?= python3

.PHONY: test preflight audit prepare render norm smoke smoke-full train-smoke export

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

preflight:
	PYTHONPATH=src scripts/check_environment.sh

audit:
	PYTHONPATH=src scripts/validate_dataset.sh --decode-videos

prepare:
	PYTHONPATH=src scripts/prepare_dataset.sh

render:
	PYTHONPATH=src scripts/render_configs.sh

norm:
	PYTHONPATH=src scripts/compute_norm_stats.sh

smoke:
	PYTHONPATH=src scripts/smoke_loader.sh

smoke-full:
	PYTHONPATH=src scripts/smoke_full_sample.sh

train-smoke:
	PYTHONPATH=src scripts/train_smoke.sh

export:
	scripts/export_code.sh /tmp/LingBot-VLA-v2-Custom-Finetune.tar.gz
