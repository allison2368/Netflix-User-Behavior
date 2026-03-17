![Build](https://github.com/allison2368/Netflix-User-Behavior/actions/workflows/lint.yml/badge.svg)
[![codecov](https://codecov.io/github/allison2368/Netflix-User-Behavior/branch/main/graph/badge.svg?token=21YX5MVR5Q)](https://codecov.io/github/allison2368/Netflix-User-Behavior)

 <img width="400" height="160" alt="image" src="https://github.com/user-attachments/assets/e02f2739-74d0-4555-b18a-79e57d62173d" />

# Netflix-User-Behavior

Our project introduces an interactive tool designed to support the Netflix platform. Aimed at marketing strategists, product managers, and content executives, the tool helps teams understand subscriber behavior, monitor engagement, and make informed decisions about content and features.

## Team Members

- Allison Peng
- Urvashi Jha
- Fatima Fazil
- Saeah Go

## Project Type

- A business visualization tool

## Table of Contents

- [Questions of Interest](#questions-of-interest)
- [Our Goal](#our-goal)
- [Data Sources](#data-sources)
- [Software Dependencies and License Information](#software-dependencies-and-license-information)
- [Repository Structure](#repository-structure)
- [Guide for Using the Tool](#guide-for-using-the-tool)
- [Project Demo](#project-demo)

## Questions of Interest

- Determining Netflix retention based on watch history, search logs, recommendations, reviews, user information.
- Which customers are higher risk of cancelling?
- Which customers should receive a retention offer?

## Our Goal

- Create a visual tool that easily answers potential business questions above.
- Work with large, messy data and understand its tradeoff.
- We will start from analyzing user behavior with Netflix data, and if possible, compare subscriptions and movies/shows with different platforms.

### Specific goals

We have three main use cases for our app, each use case corresponds to a different industry role.

1. Marketing Manager: Identify and export high-risk churners for targeted discounts
2. Studio Executive: Optimizing content investment using movie ratings
3. Product Manager: Feature engagement and churn correlation analysis

## Data Sources

### 1. Kaggle Netflix User Behavior

A large synthetic dataset with over 210,000 records capturing user interactions such as watch history, search logs, recommendation activity, and reviews. It enables behavioral analysis, feature engineering, and churn prediction.

Link: [View Dataset](https://www.kaggle.com/datasets/sayeeduddin/netflix-2025user-behavior-dataset-210k-records/data?select=watch_history.csv)

### 2. Movie API

Movie ratings were supplemented using the OMDb API; missing ratings for certain synthetic titles were imputed by genre to ensure a complete dataset for analysis.

Link: [View Dataset](https://www.omdbapi.com/)

## Software Dependencies and License Information

The project is built using Python 3.0+ and several open-source Python packages such as pandas, NumPy, scikit-learn, and Streamlit. The complete list of dependencies can be found in environment.yml. This project is licensed under the MIT License, with full details available in LICENSE.txt.

### Virtual Environment Setup

This project uses a virtual environment to ensure all collaborators run the same Python version and dependencies. After cloning the repository, create a new environment and activate it. Further instructions for running our app locally are in the examples folder.

#### 1. Create a new environment

```bash
conda env create -f environment.yml
```

#### 2. Activate the environment

```bash
conda activate netflix-env
```

#### 3. Deactivate environment after finished running app

```
conda deactivate
```

## Repository Structure

```
Netflix-User-Behavior/
├── .github/
│   └── workflows/
│       └── lint.yml
├── churn_model/
│   ├── churn_model.ipynb
│   └── churn_model_updated.py
├── docs/
│   ├── Component-Specification.md
│   ├── Functional-Specification.md
│   ├── Milestones.md
│   ├── Presentation.pdf
│   ├── Images/
│   └── technology_review/
│       ├── demo/
│       └── technology_review.md
├── examples/
│   └── readme.md
├── tests/
│   └── unit/
│       ├── test_model_usecase1.py
│       ├── test_use_case_2.py
│       ├── test_usecase3.py
│       └── test_usecase3_app.py
├── usecase1/
│   ├── usecase1.py
│   └── usecase1_app.py
├── usecase2/
│   ├── usecase2.py
│   └── usecase2_app.py
├── usecase3/
│   ├── usecase3.py
│   └── usecase3_app.py
├── dashboard.py
├── environment.yml
├── landing.py
├── pyproject.toml
├── pytest.ini
├── requirements.txt
└── styles.py
```

## Guide for Using the Tool

Each of the tabs corresponds to a different use case. Switch tabs to gain a new perspective about the netflix data!

## Code Coverage and Unit tests

To check our passing pytest unit tests, run:

```bash
pytest tests/
```

All tests should pass

Unit test code coverage: 76% across all backend .py code

## Project Demo

View the demo video for a walkthrough of the project.
