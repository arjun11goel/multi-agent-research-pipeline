
from pathlib import Path

ROOT = Path(".")
IGNORE = {
    ".venv",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "node_modules",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
}

def print_tree(path: Path, prefix: str = ""):
    entries = sorted(
        (
            item for item in path.iterdir()
            if item.name not in IGNORE
            and item.name != ".DS_Store"
        ),
        key=lambda item: (item.is_file(), item.name.lower()),
    )

    for index, item in enumerate(entries):
        last = index == len(entries) - 1
        print(prefix + ("└── " if last else "├── ") + item.name)

        if item.is_dir():
            print_tree(
                item,
                prefix + ("    " if last else "│   "),
            )

if __name__ == "__main__":
    print(ROOT.resolve().name)
    print_tree(ROOT)
