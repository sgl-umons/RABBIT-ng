<div align="center">


<img src="https://github.com/sgl-umons/rabbit/blob/main/logo.jpeg?raw=true"
       alt="RABBIT logo"
       width="400" />

[![Tests](https://github.com/sgl-umons/RABBIT/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/sgl-umons/RABBIT/actions/workflows/test.yml)
[![Last commit](https://badgen.net/github/last-commit/sgl-umons/RABBIT)](https://github.com/sgl-umons/RABBIT/commits/)
[![](https://img.shields.io/github/v/release/sgl-umons/RABBIT?label=Latest%20Release)](https://github.com/natarajan-chidambaram/RABBIT/releases/latest)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/natarajan-chidambaram/RABBIT/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/natarajan-chidambaram/RABBIT)

# RABBIT - Activity-Based Bot Identification Tool
</div>

---

## Overview

**RABBIT** is a machine-learning based tool designed to identify bot accounts among GitHub contributors .
Unlike tools that rely on profile metadata, RABBIT analyzes **behavioral activity sequences** to compute 38 distinct features.

RABBIT is developed by the **Software Engineering Lab (SGL)** at the **University of Mons (UMONS)**, Belgium.

**Why RABBIT?**

* **Behavioral Analysis:** Classifies users based on interaction timing, repository switching patterns, and activity diversity, rather than just static account details.
* **High Efficiency & Scalability:** RABBIT is designed for large-scale mining. Thanks to its incremental early-stopping mechanism, 
it can predict **thousands of accounts per hour** without reaching GitHub's imposed API rate limit (5,000 queries/hour for authorized users).


## Table of content

- [Overview](#overview)
- [Installation](#installation)
- [CLI usage](#cli-usage)
  - [Configuration (API Key)](#configuration-api-key)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
- [Python Library usage](#python-library-usage)
  - [Default usage](#default-usage)
  - [Offline usage](#offline-usage-without-api-calls)
- [How it Works](#how-it-works)
- [Citation](#citation)
- [Contributions](#contributions)
  - [Development environment setup](#development-environment-setup)
- [Authors & Credits](#authors--credits)
- [License](#license)

---

## Installation

RABBIT requires at least **Python 3.11** and can be used either as a command-line interface (CLI) tool or as a Python library.

### Option A: Using [uv](https://docs.astral.sh/uv/) 
This installs RABBIT in an isolated environment, keeping your system clean.
```shell
$ uv tool install rabbit # As a CLI tool
$ uv add rabbit          # As a Python library (to use in your uv environment)
```

### Option B: Using pip in a virtual environment 
It's recommended to use a virtual environment to avoid conflicts with other packages.
```shell
# Create and activate a virtual environment
$ python3 -m venv rabbit-env
$ source rabbit-env/bin/activate  # On Windows use `rabbit-env\Scripts\activate`
# Install RABBIT
$ pip install rabbit
```

### Option C: Using Nix (only for CLI tool)
RABBIT is also available via [Nix](https://search.nixos.org/packages?channel=unstable&show=rabbit&from=0&size=50&sort=relevance&type=packages&query=rabbit)
```shell
$ nix-shell -p rabbit
```

## CLI Usage

### Configuration (API Key)

To execute **RABBIT** for many contributors (if more than 60 API queries are required per hour), 
you need to provide a *GitHub personal access token* (API key). You can follow the instructions [here](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token) to obtain such a token.

#### Option A: Environment Variable (Recommended)  
Set the `GITHUB_TOKEN` environment variable to your GitHub personal access token.
```shell
$ export GITHUB_API_KEY=your_token_here # On Linux/Mac
$ setx GITHUB_API_KEY "your_token_here" # On Windows
```
You can also create a `.env` file in your working directory with the following content:
```text
GITHUB_API_KEY=your_token_here
```

#### Option B: Command-Line Argument
You can also provide the API key directly when running RABBIT using the `--key` argument.
```shell
$ rabbit --key your_token_here <other_arguments>
```

### Command Line Interface (CLI)

Run RABBIT command in your terminal:

<details>
<summary>Click to view <code>rabbit --help</code></summary>

```shell
$ rabbit --help
Usage: rabbit [OPTIONS] [CONTRIBUTORS]...                                                                       
                                                                                                                 
Identify bot contributors based on their activity sequences in GitHub.                                          
                                                                                                                 
╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────╮
│   contributors      [CONTRIBUTORS]...  Login names of contributors to analyze.                                │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                   │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Inputs ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --input-file  -i      FILE  Path to a file containing login names (one per line).                             │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Configuration ───────────────────────────────────────────────────────────────────────────────────────────────╮
│ --key             -k      TEXT                       GitHub API key (either in command line or in             │
│                                                      GITHUB_API_KEY env variable).                            │
│                                                      [env var: GITHUB_API_KEY]                                │
│ --min-events              INTEGER RANGE [1<=x<=300]  Min number of events required. [default: 5]              │
│ --min-confidence          FLOAT RANGE [0.0<=x<=1.0]  Confidence threshold to stop querying. [default: 1.0]    │
│ --max-queries             INTEGER RANGE [1<=x<=3]    Max API queries per contributor. [default: 3]            │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --features                      Display computed features for each contributor.                               │
│ --format    -f      [text|csv]  Format of the output. [default: text]                                         │
│ --verbose   -v      INTEGER     Increase verbosity level (can be used multiple times. -v or -vv).             │
│                                 [default: 0]                                                                  │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
</details>

#### Examples

**1 - Simple example**  
You can provide the contributor login names as positional arguments or in an input file. (Can be combined.)
```shell
$ rabbit octocat renovate 
CONTRIBUTOR                     TYPE        CONFIDENCE
tensorflow-jenkins              Bot              0.838
natarajan-chidambaram           Human            0.939
```

**2 - Export results to CSV file**
```shell
$ rabbit tensorflow-jenkins --input-file logins.txt --format csv > results.csv
```

**3 - Export with feature values**
```shell
$ rabbit --input-file logins.txt --features > results_with_features.txt
```

**4 - Increase verbosity level**  
By default, only **critical** error messages are shown.
```shell
$ rabbit --input-file logins.txt -v  # Show info and warning messages
$ rabbit --input-file logins.txt -vv # Show debug messages
```

## Python Library usage

### Default usage
The main function to use is `run_rabbit` which is an iterator yielding result for each contributor one by one.
```python
from rabbit import run_rabbit 
from dotenv import load_dotenv
import os


load_dotenv()  # Load GITHUB_API_KEY from .env file if present
API_KEY = os.getenv('GITHUB_API_KEY')

for result in run_rabbit(
    contributors=['MrRose765', 'github-actions[bot]'],
    api_key=API_KEY,
    min_events=5,
    min_confidence=1.0,
    max_queries=3,
):
    # Each result is an ContributorResult object with 'contributor', 'type', 'confidence', and 'features' attributes
    print(f"{result.contributor}: {result.user_type} (Confidence: {result.confidence})")
# Output:
# MrRose765: Human (Confidence: 0.987)
# github-actions[bot]: Bot (Confidence: 1.0)
```

### Offline usage (without API calls)
You can also use RABBIT on events data you have already collected, without making any API calls.

In that case, you need to provide a list of events for each contributor as input and write a custom function to use RABBIT:
```python
from rabbit.predictor import ONNXPredictor, predict_user_type

events = {
    'MrRose765': [ 
        # List of event dictionaries for MrRose765 ONLY
    ],
    'testuser': [ 
        # List of event dictionaries for testuser ONLY
    ],
}

# Load the pre-trained model
predictor = ONNXPredictor() # Default model path is used, you can provide a custom path if needed.

for contributor, user_events in events.items():
    result = predict_user_type(
        username=contributor,
        events=user_events,
        predictor=predictor,
    )
    print(f"{contributor}: {result.user_type} (Confidence: {result.confidence})")
# Output:
# MrRose765: Human (Confidence: 0.987)
# testuser: Bot (Confidence: 0.912)
```

## How it works
RABBIT follow a strict decision pipeline to classify a GitHub contributor that aims to minimize the number of API queries used.
### The classification pipeline
1.  **Validation & Existence Check**  
    RABBIT first verifies if the login exists on GitHub.
    * If the user does not exist: Returns `Invalid`.

2.  **Metadata Filtering (Fast Check)**  
    Before running complex analysis, RABBIT checks the account type provided by the GitHub Users API.
    * If the type is `Organization` or `Bot` (e.g., GitHub Apps): It returns this type immediately without further analysis.
    * If the type is `User`: It proceeds to the behavioral analysis.

3.  **Event Extraction**  
    RABBIT fetches the latest public events using the GitHub Events API.
    * If the number of events is below the threshold (default: 5): Returns `Unknown` (Insufficient data).

4.  **Feature Extraction**  
    The retrieved events are converted into activity sequences (using the [ghmap](https://github.com/sgl-umons/ghmap) tool.   
    RABBIT computes **38 behavioral features** covering volume, timing (inter-arrival time), and switching patterns (between repositories and activity types).

5.  **Prediction (BIMBAS Model)**  
    The computed features are fed into the machine learning model (Gradient Boosting).
    * Returns: `Human` or `Bot`.
    * Confidence Score: A value between 0.0 and 1.0 indicating the certainty of the prediction.
      
### ⚠️ Limitations & Accuracy

RABBIT is based on a probabilistic machine learning model trained on a ground-truth dataset. While it achieves high accuracy, it is **not infallible**.

* **Misclassifications:** It is possible for a Human to be classified as a Bot (or vice versa), especially if their activity pattern is highly repetitive or unusual.
* **Data Scarcity:** Accounts with very few public events are harder to classify. The tool defaults to `Unknown` to avoid guessing when data is scarce.

If you encounter a clear misclassification, please open an issue on GitHub so we can investigate and improve the model.

---

## Citation
This tool was developed as part of the research work by Natarajan Chidambaram, Tom Mens and Alexandre Decan. 
It is part of a research article titled "A Bot Identification Model and Tool based on GitHub Activity Sequences"
([doi](https://doi.org/10.1016/j.jss.2024.112287))

If you use RABBIT in your research, please cite it using the following BibTeX entry:
```bibtex
@article{Chidambaram_RABBIT_A_tool,
author = {Chidambaram, Natarajan and Mens, Tom and Decan, Alexandre},
doi = {10.1145/3643991.3644877},
title = {{RABBIT: A tool for identifying bot accounts based on their recent GitHub event history}}
}
```

## Contributions
Contributions to RABBIT are welcome! If you encounter any issues or have suggestions for improvements, 
please open an issue or submit a pull request directly on GitHub.

When contributing, please ensure that your code adheres to the existing coding style and includes appropriate tests.  
Also, make sure to clearly document **why** the changes are necessary in the commit messages and pull request descriptions.

### Development environment setup
We use `uv` for managing the development environment.
```shell
# Clone the repository (must be your fork if you plan to contribute)
$ git clone https://github.com/sgl-umons/RABBIT.git
$ cd RABBIT

# Install development dependencies
$ uv sync --dev

# Run tests
$ uv run pytest
# Lint the code
$ uv run ruff check
# Format the code
$ uv run ruff format 
```

## Authors & Credits
This tool is maintained by the **Software Engineering Lab (SGL)** of the **University of Mons (UMONS)**, Belgium.

### Contributors:
- [**Natarajan Chidambaram**](https://github.com/natarajan-chidambaram) (Original author)
- [**Alexandre Decan**](https://github.com/alexandredecan)
- [**Tom Mens**](https://github.com/tommens)
- [**Cyril Moreau**](https://github.com/MrRose765)

## License
This tool is distributed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). See the [LICENSE](LICENSE) file for more details.
