.PHONY: clean download eval eval_example example_zip features features_example lint requirements predict predict_example predict_submission shards shards_example test_environment train train_example

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = conferencing-speech-2022
PYTHON_VERSION = 3.9
UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
PYTHON_INTERPRETER = python3
ACTIVATE_CMD = "source venv/bin/activate"
endif
ifeq ($(UNAME), Windows)
PYTHON_INTERPRETER = python
ACTIVATE_CMD = "source venv/Scripts/activate"
endif



#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
# 	ipython kernel install --user --name=$(PROJECT_NAME)
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install ipykernel
	python -m ipykernel install --user --name $(PROJECT_NAME)
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html

## Make Dataset
download: requirements
	$(PYTHON_INTERPRETER) src/data/make_download.py

## Make Features
features: download
	$(PYTHON_INTERPRETER) src/data/make_features.py

## Make Features (example)
features_example: download
	$(PYTHON_INTERPRETER) src/data/make_features.py --example

## Make Example ZIP
example_zip: features_example
	$(PYTHON_INTERPRETER) src/data/make_example_zip.py

## Make Shards
shards: features
	$(PYTHON_INTERPRETER) src/data/make_shards.py

## Make Shards (example)
shards_example: features_example
	$(PYTHON_INTERPRETER) src/data/make_shards.py --example

## Make Norm
norm: features
	$(PYTHON_INTERPRETER) src/data/make_norm.py

## Make Norm (example)
norm_example: features_example
	$(PYTHON_INTERPRETER) src/data/make_norm.py --example

## Do training
train: shards
	$(PYTHON_INTERPRETER) src/train/make_train.py

## Do training with example data
train_example: shards_example
	$(PYTHON_INTERPRETER) src/train/make_train.py --example

## Do prediction
predict: train
	$(PYTHON_INTERPRETER) src/predict/make_predict.py

## Do prediction for submission
predict_submission: train
	$(PYTHON_INTERPRETER) src/predict/make_predict_final_submission.py

## Do training with example data
predict_example: train_example
	$(PYTHON_INTERPRETER) src/predict/make_predict.py --example

## Do evaluation
eval: predict
	$(PYTHON_INTERPRETER) src/eval/make_eval.py

## Do evaluation (example)
eval_example: predict_example
	$(PYTHON_INTERPRETER) src/eval/make_eval.py --example


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Set up python interpreter environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv ./venv/
	@echo ">>> Virtual environment created under venv/."

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
