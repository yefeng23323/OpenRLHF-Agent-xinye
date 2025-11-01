import os
import sys
import platform
from datetime import datetime
from setuptools import find_packages, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

_build_mode = os.getenv("OPENRLHF_AGENT_BUILD_MODE", "")


def _is_nightly():
    return _build_mode.lower() == "nightly"


def _fetch_requirements(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fd:
            return [r.strip() for r in fd.readlines() if r.strip() and not r.startswith("#")]
    return []


def _fetch_readme():
    readme_path = "README.md"
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return "OpenRLHF Agent."


def _fetch_version():
    version_file = "version.txt"
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            version = f.read().strip()
    else:
        version = "0.1.0"

    if _is_nightly():
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        version += f".dev{date_str}"

    return version


def _fetch_package_name():
    return "openrlhf-agent-nightly" if _is_nightly() else "openrlhf-agent"


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        abi_tag = f"{python_version}"

        if platform.system() == "Linux":
            platform_tag = "manylinux1_x86_64"
        else:
            platform_tag = platform.system().lower()

        return python_version, abi_tag, platform_tag

setup(
    author="freder",
    name=_fetch_package_name(),
    version=_fetch_version(),
    package_dir={"": "src"},
    packages=find_packages(
        where="src",
        exclude=(
            "tests",
            "docs",
            "examples",
            "*.tests",
            "*.tests.*",
            "tests.*",
        )
    ),
    description="A RLHF agent framework built on OpenRLHF.",
    long_description=_fetch_readme(),
    long_description_content_type="text/markdown",
    install_requires=[],
    extras_require={
        "vllm": ["vllm>=0.10.2"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: System :: Distributed Computing",
    ],
    keywords=[
        "reinforcement learning",
        "agent",
    ],
    project_urls={
        "Homepage": "https://github.com/OpenRLHF/openrlhf-agent",
        "Bug Reports": "https://github.com/OpenRLHF/openrlhf-agent/issues",
        "Source": "https://github.com/OpenRLHF/openrlhf-agent",
        "Documentation": "https://openrlhf-agent.readthedocs.io/",
    },
    cmdclass={"bdist_wheel": bdist_wheel},
    include_package_data=True,
    zip_safe=False,
)
