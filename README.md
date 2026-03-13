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

- Questions of Interest
- Our Goal
- Data Sources
- Software Dependencies and License Information
- Directory Summary
- Guide for Using the Tool
- Project Demo

## Questions of Interest

- Determining Netflix retention based on watch history, search logs, recommendations, reviews, user information.
- Which customers are higher risk of cancelling?
- Which customers should receive a retention offer?

## Our Goal

- Create a visual tool that easily answers potential business questions above.
- Work with large, messy data and understand its tradeoff.
- We will start from analyzing user behavior with Netflix data, and if possible, compare subscriptions and movies/shows with different platforms.

### Specific goals for Churn Analysis

- Define the Churn Event: Determine what constitutes a churned user (e.g., cancellation of subscription, non-renewal).
- Calculate the Rate: Calculate the percentage of customers lost over a specific period.
- Segment Customer Data: Group customers by behavior, demographics, or subscription type to identify which groups are leaving.
- Identify Patterns: Analyze usage data for trends, such as declining activity or lack of product engagement.
- Act on Insights: Create targeted campaigns or product improvements to address the specific reasons for attrition. 

## Data Sources

### 1. Netflix Customer Subscription Dataset

Contains information about user subscription plans, billing details, and account status. This dataset helps analyze subscription behavior, user demographics, and potential indicators of churn.

Link: [View Dataset](https://www.kaggle.com/datasets/sureshmuthusamy001p/netflix-customer-subscription)

### 2. Netflix Movies and TV Shows Dataset

Provides metadata about Netflix content, including titles, genres, release year, cast, director, and content type. This dataset supports analysis of content characteristics and their relationship with user engagement and retention.

Link: [View Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)

### 3. Netflix 2025 User Behavior Dataset

A large synthetic dataset with over 210,000 records capturing user interactions such as watch history, search logs, recommendation activity, and reviews. It enables behavioral analysis, feature engineering, and churn prediction.

Link: [View Dataset](https://www.kaggle.com/datasets/sayeeduddin/netflix-2025user-behavior-dataset-210k-records/data?select=watch_history.csv)

### 4. Movie API

Movie ratings were supplemented using the OMDb API; missing ratings for certain synthetic titles were imputed by genre to ensure a complete dataset for analysis.

## Software Dependencies and License Information

The project is built using Python 3.0+ and several open-source Python packages such as pandas, NumPy, scikit-learn, and Streamlit. The complete list of dependencies can be found in requirements.txt. This project is licensed under the MIT License, with full details available in LICENSE.txt.

## Directory Summary

/examples
  # Example usage of the project, including demonstration images and workflows

/box_office_prediction
  # Main project module containing core code
  /notebooks
    # Jupyter notebooks for data cleaning, ML, and feature engineering
  /models
    # Saved machine learning models
  /tests
    # Unit and edge tests for ML pipeline and Streamlit app
  /__pycache__
    # Python cache files (auto-generated)

/data
  /cleaned
    # Preprocessed datasets ready for analysis
  /raw
    # Raw datasets as downloaded from Kaggle 

/docs
  # Documentation including functional specification, component specification, milestones, and technology reviews.

/scripts
  # Scripts for preprocessing, feature engineering, and utility functions
  /__pycache__
    # Python cache files (auto-generated)

/app
  # Streamlit application code for interactive dashboards

/setup.py or pyproject.toml
  # Project installation and packaging configuration

README.md
  # Project overview, instructions, and functional specification

## Guide for Using the Tool

## Project Demo
View the demo video for a walkthrough of the project. 



