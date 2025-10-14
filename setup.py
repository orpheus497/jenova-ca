from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="jenova-ai",
    version="3.1.0",
    author="orpheus497",
    description="Jenova Cognitive Architecture with Cognitive Process Accelerator (CPA) - A truly 'alive' AI with persistent state management, proactive cognitive engagement, and enterprise-grade performance optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/orpheus497/jenova-ai",
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