"""Configuration for the fairness-aware calibration experiments."""
import os
import random
import numpy as np

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-4.1"  # Primary evaluator model

# Experiment parameters
N_SAMPLES = 100        # Number of base text samples
N_RUNS = 3             # Repetitions per condition for reliability
TEMPERATURE = 0.3      # Low temperature for consistency, >0 for variance measurement
MAX_TOKENS = 500       # Max tokens for evaluation response

# Disclosure text variants
DISCLOSURE_TEXT = "Note: This response was written with AI assistance."
DEMOGRAPHIC_NATIVE = "Note: The author is a native English speaker."
DEMOGRAPHIC_NONNATIVE = "Note: The author is a non-native English speaker."

# Calibration prompt additions
FAIRNESS_INSTRUCTION = (
    "IMPORTANT: Evaluate the content quality ONLY based on the rubric criteria below. "
    "Do NOT consider whether the author used AI tools, the author's language background, "
    "or any other non-content factors. Focus exclusively on the substance and quality of the response."
)

EVIDENCE_FIRST_INSTRUCTION = (
    "You MUST first provide detailed evidence for each rubric criterion before assigning a score. "
    "List specific strengths and weaknesses in the response text. "
    "Only after completing your evidence analysis should you assign a final score."
)
