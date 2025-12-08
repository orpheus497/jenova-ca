import datetime
import subprocess
import shlex

def get_current_datetime() -> str:
    """
    Returns the current date and time in ISO 8601 format.
    """
    return datetime.datetime.now().isoformat()

def execute_shell_command(command: str, description: str) -> dict:
    """
    Executes a shell command and returns the result.
    """
    try:
        # Use shlex.split to safely parse the command string
        command_args = shlex.split(command)
        result = subprocess.run(command_args, capture_output=True, text=True, check=False)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "error": result.returncode != 0
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "error": True
        }