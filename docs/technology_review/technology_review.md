# Technology Review

# 1. Data Processing Technology Review

## 1.1 Background and Use Case

**Problem:** We need an effective python library that allows us to perform data manipulation through dataframes-based analysis. There are two commonly used libraries, pandas and polars. Pandas has been historically used for dataframe-based analysis, however polars was later introduced and known to be more computationally efficient than pandas. We want to figure out which library allows us to best analyze over 130,000 observations over 6 relational data tables. 

**Use Case:** We need to efficiently clean, transform and analyze structured datasets that contain categorical and numerical variables. Specifically, this includes loading in datasets from Google Bigquery, aggregating metrics across groups, or performing data cleaning. We also need a dataframe-based library that integrates well with other libraries in Python, including feature engineering or data visualization.

## 1.2 Python Package Choices and Comparisons

**Pandas:** Popular data science tool created in 2008 by McKinney, originally created for high-performance financial data analysis. 

**Features:**

- Single threaded 
- Uses numpy arrays 
- More supported ecosystem with other libraries. Use when you are integrating a wide range of existing python packages
- Generally, more data scientists are familiar with pandas syntax
- Muttable dataframes 
- Better for small to medium sized datasets that don’t need scaling


**Polars:** Newer data science tool started in 2020 by Ritchie Vink and was created because of the lack of mature dataframe processing languages made with Rust. It uses a multithread dataframe library, compared to pandas only uses one.   

**Features:**

- Multi threaded 
- Uses apache arrow memory format
- More memory efficient
- Faster with larger databases, runs query optimization before execution
- Immutable dataframes 
- Better for scaling data operations, use when working with large datasets where performance is important 
- Use Polars if analysis requires repetitive computations and can utilize lazy execution

## 1.3 Final Choice and Potential Drawbacks 

Our final choice is to use the Pandas dataframe. Even though the pandas dataframes are loaded in slower and have less computational power compared to polars, the scale of our dataset does not require a largely efficient library at the moment. At our scale, the difference of time is currently at tenths of a second, which will not make a difference for our data visualization dashboard. In addition, we need the ability to integrate the pandas dataframe with our feature engineering tool of choice: feature tools. The pandas dataframe aligns with the feature tools library and will make the bulk of our project easier as we are hoping to focus more on the feature engineering side rather than data manipulation. A potential drawback is if we do want to eventually scale our data or work with live data visualization. Using a pandas library will be computationally expensive, however it is easy to convert our panda dataframes to arrow tables.  

# 2. Feature Engineering Technology Review

## 2.1 Background and Use Case

**Problem:** We need an automated feature engineering library that can generate predictive features from our Netflix dataset for churn prediction. Our dataset consists of 6 relational tables (users, movies, watch history, recommendations, search logs, reviews) with over 130,000 events. Manually engineering features from this structure is time-consuming and risks missing important signals. We need a library that can automate this process efficiently and produce interpretable, model-ready features.

**Use Case:** The feature engineering phase of our Netflix churn prediction pipeline requires transforming 6 relational tables into a single flat feature matrix (one row per user) that a classification model can consume. Each user must be represented by numerical summaries of their behavior across watch history, search activity, recommendation engagement, and review patterns. The library we choose must handle multi-table relational joins, aggregate event-level data to user level, and produce features that are both predictive and interpretable enough to explain to stakeholders why a user is flagged as churn risk.

**Requirements:**

- Must aggregate event-level rows (watch, search, reviews) up to one row per user.
- Must handle foreign key relationships between 6 tables without manual join logic.
- Must produce named, interpretable features suitable for stakeholder reporting.
- Must scale to 130,000+ event rows without prohibitive runtime.
- Must be open-source with active maintenance and Python 3.10+ support.

## 2.2 Feature Engineering Library Review

### Featuretools

**Author:** James Max Kanter and Kalyan Veeramachaneni, MIT CSAIL. Originally released 2015 by Feature Labs.

- Built specifically for relational datasets using a technique called Deep Feature Synthesis (DFS). DFS traverses relationships between tables and automatically constructs features by applying aggregation and transformation primitives across the table hierarchy. 
- Generates human-readable feature names making it straightforward to explain features to non-technical stakeholders.
- Can be memory intensive, generate low-signal features that require post-generation feature selection and can lead to feature explosion.
- Parallel processing via Dask and multiprocessing.
- Actively maintained.

### tsfresh

**Author:** Maximilian Christ, Nils Braun, Julius Neuffer, and Andreas Kempa-Liehr. Released 2016 by Blue Yonder GmbH.

- Open-source time-series feature extraction library designed to automatically extract a large number of statistical features from time-series data but does not natively handle multi-table data.
- Parallel processing built-in.
- Well suited for detecting temporal decay patterns. e.g. declining watch frequency in the weeks before churn.
- Actively maintained.

## 2.3 Final Choice

Featuretools is the stronger choice for our dataset structure. Our 6 relational tables connected by user_id and movie_id foreign keys map directly to Featuretools EntitySet. A single DFS call will automatically generate user-level aggregations across all tables: total watches, average progress, search click rate, review sentiment, recommendation engagement and more. These are precisely the features most likely to distinguish churned from active users. 

# 3. Machine Learning Framework Technology Review

## 3.1 Background and Use Case

**Problem:** Subscription-based platforms like Netflix often struggle with unpredictable customer churn. If high-risk users are not identified early, the business loses recurring revenue, which can significantly impact growth. The primary goal of this project is to build a system that predicts which users are likely to cancel their subscription. With early detection, the marketing team can proactively target these users with retention campaigns or special offers.

**Use Case:** Marcus, the Marketing Manager, aims to identify users who have a high probability of churn, specifically those exceeding an 85% likelihood, so that they can be targeted with personalized discount campaigns. To achieve this, the system uses aggregated user-level features derived from multiple behavioral datasets, including watch history, search logs, recommendation logs, reviews, and movie ratings. The output consists of churn predictions along with insights, highlighting which specific user behaviors contribute most to the risk of churn, enabling Marcus and his team to make informed retention decisions.

**Rationale for using a Python Library:** Building machine learning models from scratch for hundreds of features is both time-consuming and prone to errors. Python machine learning libraries address these challenges by providing efficient, pre-built tools that streamline development and lets you experiment rapidly. Additionally, these libraries support visualization and interpretability, which are essential for explaining predictions and insights to stakeholders.

## 3.2 Python Package Choices 

### Scikit-learn

- **Author:** Developed by INRIA, first released in 2010.
- **Summary:** Provides ready-to-use, optimized machine learning algorithms for structured/tabular data, including Random Forest Classifiers.
- **Strengths:** Easy to use, highly interpretable, requires minimal preprocessing, and provides feature importance metrics.

### TensorFlow

- **Author:** Developed by Google Brain, first released in 2015.
- **Summary:** A flexible framework for building neural networks and deep learning models.
- **Strengths:**  Supports large-scale computations, provides probabilistic predictions, and allows future scalability into complex deep learning architectures.
- **Considerations:** Requires careful preprocessing, tuning, and setup, especially on Windows platforms.

## 3.3 Package Comparisons 

| Aspect                   | Scikit-learn RF                  | TensorFlow NN                              |
|--------------------------|----------------------------------|--------------------------------------------|
| Accuracy                 | 0.86                             | 0.58                                       |
| Feature importance       | Yes                              | No (not directly)                          |
| Preprocessing            | Minimal                          | Requires scaling                           |
| Class imbalance handling | Easier                           | Requires tuning                            |
| Interpretability         | High                             | Low                                        |
| Complexity               | Low                              | Higher                                     |
| Setup                    | Simple                           | More demanding (OS/library dependencies)   |

The Scikit-learn Random Forest model performed strongly, achieving 86% accuracy. Its built-in feature importance clearly identified key churn drivers such as total watch time, number of sessions, and average movie rating, making the results easy for stakeholders to understand and act on. Installation was straightforward with pip install scikit-learn, requiring no additional system dependencies.

In comparison, the TensorFlow Neural Network achieved 58% accuracy and produced churn probabilities rather than direct feature importance. It also required more setup and tuning. Because the demo used Python 3.12.6, which is not fully supported by standard TensorFlow builds, installation initially failed due to DLL and pywrap_tensorflow errors. This was resolved by installing the Microsoft Visual C++ Redistributable and creating a dedicated virtual environment (venv_tf).

## 3.4 Final Choice  

**Selected Library:** Scikit-learn Random Forest

**Reasons for Selection:**

- Higher predictive accuracy with minimal preprocessing.
- Easily interpretable outputs via feature importance, enabling actionable insights.
- Simple setup and lower computational complexity.
- Well-suited for tabular business datasets.

While TensorFlow provides flexibility and future-proofing for deep learning, for this use case, Scikit-learn's Random Forest classifier offers a balance of accuracy, interpretability, and ease-of-use that meets business needs effectively.

## 3.5 Drawbacks  

Scikit-learn’s Random Forest worked well on our aggregated dataset, but there are considerations for full-scale deployment. Because it runs in-memory on a single machine, training time and memory usage could increase significantly with very large datasets. In a production setting, computational efficiency would therefore depend heavily on available cloud resources. Also, while feature importance makes the model easier to interpret, it doesn’t prove causal relationships, so any business decisions based on it should be validated further.

Neural networks can capture more complex interactions between features, but for structured tabular business data, tree-based models like Random Forest are often just as effective, easier to work with, and more reliable.

# 4. Dashboard Framework Technology Review

## 4.1 Background and Use Case

**Problem:** We want to give non-technical users (Marcus and Sarah) an interactive dashboard to explore customer churn and content engagement.

**Use Cases:**

**Use Case 1 (Marcus):** Wants to identify users with >85% churn probability and export them for a discount campaign.

**Use Case 2 (Sarah):** Wants to see if high-rated content (IMDb 8+) improves long-term retention.

**Requirements for the technology:**

1. **Interactive visualizations:** Users have filters and see predictions or charts right away.
2. **Data export:** Users should be able to download results as CSV
3. **Ease of use:** Intuitive UI without confusion
4. **Compatibility:** Since we use Python for this project it should work with Python 3 and integrate with existing ML models (customer segmentation, churn prediction).

## 4.2 Python Package Choices and Comparisons

**Streamlit:** A Python library developed by Streamlit Inc. for quickly building web apps in 2019. Really simple to set up and use, allowing developers to create interactive dashboards with minimal code.

- Easy to learn
- Simple layout
- CSV export button available with st.download_button 
- Suitable for small to medium datasets
- Less flexible layout compared to Dash 

**Plotly dash:** Dash is developed by Plotly Technologies in 2017 for building web apps with Plotly visualizations. We can make highly customizable and flexible dashboards with plotly.

- Difficult to learn compared to Streamlit
- Customizable and flexible layout
- Full integration with Plotly charts
- CSV export is supported with dcc.Download
- Suitable for large dashboards and multi-page apps

## 4.3 Final Choice

In the simple demo, we were able to do the same tasks with 100 lines of code with Streamlit, but needed 190 lines of code with plotly. Another factor we considered was users. As our primary users are non-technical stakeholders, we think it is important to have an intuitive and easy-to-use UI. Also we considered whether we can integrate easily with pretrained models. As every function we want to do already works with Streamlit, and due to our time limit (1month), we chose to use Streamlit.

## 4.4 Drawbacks and Concerns

One concern at this stage is that, due to Streamlit’s limited layout control, we may face challenges expanding the project in the future, particularly outside of a class project and in a production environment.
