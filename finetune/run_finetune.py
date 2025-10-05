
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Assuming prepare_data is in the same directory
from prepare_data import prepare_data

# Establish the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLAMA_CPP_DIR = PROJECT_ROOT / "llama.cpp"
FINETUNE_DATA_DIR = PROJECT_ROOT / "finetune"
DEFAULT_OUTPUT_FILE = FINETUNE_DATA_DIR / "finetune_train.jsonl"
MODELS_DIR = PROJECT_ROOT / "models"

def run_command(command, cwd):
    """Runs a command in a subprocess and streams its output."""
    print(f"Running command: {' '.join(command)} in {cwd}")
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        print(f"Error: Command {' '.join(command)} failed with exit code {process.returncode}")
        sys.exit(process.returncode)

def setup_llama_cpp():
    """Ensures llama.cpp is cloned and built."""
    if not LLAMA_CPP_DIR.exists():
        print("llama.cpp not found. Cloning the repository...")
        run_command(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(LLAMA_CPP_DIR)], cwd=PROJECT_ROOT)
    else:
        print("llama.cpp found. Pulling latest changes...")
        run_command(["git", "pull"], cwd=LLAMA_CPP_DIR)

    # Build llama.cpp
    print("Building llama.cpp...")
    # To build the finetune executable, we need to run make finetune
    run_command(["make", "finetune"], cwd=LLAMA_CPP_DIR)

def run_finetuning(model_path, data_file):
    """Runs the llama.cpp finetuning process."""
    print("Starting llama.cpp finetuning...")
    finetune_executable = LLAMA_CPP_DIR / "build" / "bin" / "llama-finetune"
    if not finetune_executable.exists():
        print(f"Error: finetune executable not found at {finetune_executable}")
        print("Please ensure llama.cpp is built correctly.")
        sys.exit(1)

    finetuned_model_path = FINETUNE_DATA_DIR / "finetuned-model.gguf"

    # These are example parameters, they should be made configurable
    # The command is based on the README in llama.cpp/examples/training
    command = [
        str(finetune_executable),
        "--model", str(model_path),
        "--file", str(data_file),
        "--out-model", str(finetuned_model_path),
        "-ngl", "999", # Use maximum GPU layers
        "-c", "512",
        "-b", "512",
        "-ub", "512",
    ]

    run_command(command, cwd=LLAMA_CPP_DIR)
    print(f"Finetuning complete. Finetuned model saved to {finetuned_model_path}")

def main():
    parser = argparse.ArgumentParser(description="End-to-end finetuning process for Jenova AI.")
    parser.add_argument("--model", required=True, help="Path to the base model to finetune (e.g., models/llama-3.2-1b-f32.gguf)")
    parser.add_argument("--insights-dir", default=os.path.join(os.path.expanduser("~"), ".jenova-ai", "insights"), help="Directory containing the user's insights.")
    parser.add_argument("--history-file", help="Path to a conversation history file to include.")
    args = parser.parse_args()

    print("Starting the finetuning process...")

    # 1. Setup llama.cpp
    setup_llama_cpp()

    # 2. Prepare the training data
    print("Preparing training data...")
    prepare_data(DEFAULT_OUTPUT_FILE, args.insights_dir, args.history_file)

    # 3. Run the finetuning
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    run_finetuning(model_path, DEFAULT_OUTPUT_FILE)

    print("Finetuning process completed successfully.")


if __name__ == "__main__":
    main()
