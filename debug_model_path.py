from pathlib import Path
import os

print(f"CWD: {os.getcwd()}")
search_path = Path("models")
print(f"Path('models') absolute: {search_path.absolute()}")
print(f"Exists: {search_path.exists()}")
if search_path.exists():
    print(f"Contents: {list(search_path.iterdir())}")
    gguf_files = list(search_path.glob("*.gguf"))
    print(f"GGUF files found: {gguf_files}")
