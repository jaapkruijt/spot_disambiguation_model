from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read().strip()

setup(
    name="spot.disambiguation",
    description="Repository for disambiguating mentions used during the SPOT reference game",
    version=version,
    data_files=[('VERSION', ['VERSION'])],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaapkruijt/spot_scenario_creation",
    license='MIT License',
    authors={
        "Jaap Kruijt": ("Jaap kruijt", "j.m.kruijt@vu.nl"),
        "Baier": ("Thomas Baier", "t.baier@vu.nl")
    },
    package_dir={'': 'src'},
    packages=find_namespace_packages(include=['spot.*', 'spot_service.*'], where='src'),
    package_data={},
    python_requires='>=3.7',
    install_requires=[
        'emissor',
        "cltl.combot",
        "gensim",
        "sentence_transformers",
        "spacy",
        "thefuzz",
        "torch"
    ],
    extras_require={
        "service": [
            "cltl.combot",
        ]
    },
    setup_requires=['flake8']
)
