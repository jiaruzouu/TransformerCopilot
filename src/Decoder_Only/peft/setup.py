from setuptools import find_packages, setup

extras = {}
extras["quality"] = ["black ~= 22.0", "ruff>=0.0.241"]
extras["docs_specific"] = ["hf-doc-builder"]
extras["dev"] = extras["quality"] + extras["docs_specific"]
extras["test"] = extras["dev"] + ["pytest", "pytest-xdist", "parameterized"]

setup(
    name="peft",
    version="0.3.0.dev0",
    description="Parameter-Efficient Fine-Tuning (PEFT)",
    license_files=["LICENSE"],
    keywords="deep learning",
    license="Apache",
    author="The AGI-Edgerunners team",
    author_email="agi.edgerunners@gmail.com",
    url="https://github.com/AGI-Edgerunners/LLM-Adapters",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={},
    python_requires=">=3.7.0",
    install_requires=[
        "numpy>=1.17",
        "packaging>=20.0",
        "psutil",
        "pyyaml",
        "torch>=1.13.0",
        "transformers",
        "accelerate",
    ],
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
