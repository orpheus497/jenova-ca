from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="jenova-ai",
    version="1.0.0",
    author="orpheus497",
    description="The perfected Jenova Cognitive Architecture with a dynamic Insight Engine and self-optimizing context.",
    long_description="A self-correcting, learning AI assistant built on a robust, pragmatic cognitive architecture.",
    url="https://github.com/orpheus497",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "jenova=jenova.main:main",
        ],
    },
)