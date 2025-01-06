default: | help

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "   run {model}        to run using sample dataset with saving all files"
	@echo "   dry-run {model}    to run using sample dataset without saving any files"
	@echo "   test               to run integration test"
	@echo "   check              to type check"
	@echo "   setup              to setup to run"
	@echo "   args               to see argments"

# run using sample dataset with saving any logfiles
run:
	poetry run python src/main.py config=conf/$(word 3, $(MAKECMDGOALS)).yaml dataset=$(word 2, $(MAKECMDGOALS)) save_log=true save_model=true

# run using sample dataset without saving any files
dry-run:
	poetry run python src/main.py config=conf/$(word 3, $(MAKECMDGOALS))_dev.yaml dataset=$(word 2, $(MAKECMDGOALS))

test:
	poetry run python src/main.py config=conf/peter_dev.yaml dataset=$(word 2, $(MAKECMDGOALS)) model.type_rating_embedding=null
	poetry run python src/main.py config=conf/peter_dev.yaml dataset=$(word 2, $(MAKECMDGOALS)) model.type_rating_embedding=0
	poetry run python src/main.py config=conf/peter_dev.yaml dataset=$(word 2, $(MAKECMDGOALS)) model.type_rating_embedding=1
	poetry run python src/main.py config=conf/peter_dev.yaml dataset=$(word 2, $(MAKECMDGOALS)) model.type_rating_embedding=0 ablation.leak_rating=true
	poetry run python src/main.py config=conf/pepler_dev.yaml dataset=$(word 2, $(MAKECMDGOALS)) model.type_rating_embedding=null
	poetry run python src/main.py config=conf/pepler_dev.yaml dataset=$(word 2, $(MAKECMDGOALS)) model.type_rating_embedding=0
	poetry run python src/main.py config=conf/pepler_dev.yaml dataset=$(word 2, $(MAKECMDGOALS)) model.type_rating_embedding=1
	poetry run python src/main.py config=conf/pepler_dev.yaml dataset=$(word 2, $(MAKECMDGOALS)) model.type_rating_embedding=0 ablation.leak_rating=true
	poetry run python src/main.py config=conf/erra_dev.yaml dataset=$(word 2, $(MAKECMDGOALS)) model.type_rating_embedding=null
	poetry run python src/main.py config=conf/cer_dev.yaml dataset=$(word 2, $(MAKECMDGOALS))
	poetry run python src/main.py config=conf/pepler_d_dev.yaml dataset=$(word 2, $(MAKECMDGOALS))

# type check
check:
	poetry run mypy src/main.py --ignore-missing-imports

# setup
setup:
	poetry install
	poetry run python -m spacy download en_core_web_sm

# download datsets
load:
	sh scripts/download_all.sh


# see argments
args:
	poetry run python src/main.py -h
