.PHONY: submit install clean clean-results clean-datasets clean-cache \
 use-reference use-student \
 test \
 test-b submit-b \
 test-c submit-c \
 test-d submit-d \
 test-e submit-e \
 test-f submit-f \
 test-g submit-g \
 test-h submit-h \
 test-i submit-i \
 test-j submit-j \
 submit-k submit-l submit-m submit-n submit-o submit-p \
 demo-task-j \
 demo-task-k demo-task-l demo-task-m demo-task-n demo-task-o \
 demo-task-p

# ---- Task B: Linear layer ---------------------------------------------------

test-b:
	uv run pytest tests/test_linear.py

submit-b:
	uv run python submit.py b

# ---- Task C: Activation functions -------------------------------------------

test-c:
	uv run pytest tests/test_activations.py

submit-c:
	uv run python submit.py c

# ---- Task D: Loss functions --------------------------------------------------

test-d:
	uv run pytest tests/test_loss.py

submit-d:
	uv run python submit.py d

# ---- Task E: Regularization (Dropout) ---------------------------------------

test-e:
	uv run pytest tests/test_dropout.py

submit-e:
	uv run python submit.py e

# ---- Task F: Model classes (Sequential and MLP) -----------------------------

test-f:
	uv run pytest tests/test_model.py

submit-f:
	uv run python submit.py f

# ---- Task G: Optimizers (SGD and Adam) --------------------------------------

test-g:
	uv run pytest tests/test_optim.py

submit-g:
	uv run python submit.py g

# ---- Task H: Data handling (Dataset and DataLoader) -------------------------

test-h:
	uv run pytest tests/test_data.py

submit-h:
	uv run python submit.py h

# ---- Task I: Evaluation metrics (accuracy and confusion matrix) -------------

test-i:
	uv run pytest tests/test_evaluation.py tests/test_confusion_matrix.py tests/test_parameter_metrics.py

submit-i:
	uv run python submit.py i

# ---- Task J: Training loop and learning-rate experiment ---------------------

test-j:
	uv run pytest tests/test_training.py

submit-j:
	uv run python submit.py j

demo-task-j:
	uv run python scripts/experiments.py lr_sweep \
		--seed 0 \
		--epochs 20 \
		--save results/figures/lr_experiment.png

# ---- Tasks K-P: Applied experiments ----------------------------------------

demo-task-k:
	uv run python scripts/experiments.py boundaries --seed 0

demo-task-l:
	uv run python scripts/experiments.py diagnostics --seed 0

demo-task-m:
	uv run python scripts/experiments.py regression --seed 0

demo-task-n:
	uv run python scripts/experiments.py scaling --seed 0

demo-task-o:
	uv run python scripts/experiments.py hparams --seed 0

demo-task-p:
	uv run python scripts/experiments.py weights --seed 0

submit-k:
	uv run python submit.py k

submit-l:
	uv run python submit.py l

submit-m:
	uv run python submit.py m

submit-n:
	uv run python submit.py n

submit-o:
	uv run python submit.py o

submit-p:
	uv run python submit.py p

# ---- Utilities --------------------------------------------------------------

submit:
	uv run python submit.py

test:
	uv run pytest tests/

install:
	git init
	uv sync

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

clean-results:
	find results -mindepth 1 -not -type d -not -name '.gitkeep' -delete

clean-datasets:
	find data -mindepth 1 -not -type d -not -name '.gitkeep' -delete

clean-cache:
	rm -rf results/models/

use-reference:
	@if [ -d src_student_backup ]; then \
	    echo "Already in reference mode. Run 'make use-student' first."; \
	    exit 1; \
	elif [ ! -d src_reference ]; then \
	    echo "src_reference/ not found. Run 'make compile-reference' first."; \
	    exit 1; \
	else \
	    mv src src_student_backup && \
	    cp -r src_reference src && \
	    touch .reference_mode && \
	    echo "Reference mode active. Run 'make use-student' to switch back."; \
	fi

use-student:
	@if [ ! -d src_student_backup ]; then \
	    echo "Already in student mode."; \
	else \
	    rm -rf src && \
	    mv src_student_backup src && \
	    rm -f .reference_mode && \
	    echo "Student mode active."; \
	fi
