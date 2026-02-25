# Milestones: Netflix User Behavior Project


## Milestone 1: Data Infrastructure & Data Manager Development
- Kaggle Dataset Preprocessing: Clean and preprocess Kaggle dataset using Python. 
- API Integration: Integrate OMDB/TMDB API for content metadata. 
- Data Manager API Implementation: Create a Python class/module so that the Analytics Engine and Visualization Manager can access the data easily.
- Initial EDA: Generate summary statistics of watch history, visualize genre, duration, and user engagement patterns, and analyze subscription trends and retention metrics.


## Milestone 2: Analytics & Prediction Engine (ML Pipeline)
- Feature Engineering: Set the features for churn prediction (Either add new features or select features)
- Model Training & Selection: Train machine learning model to predict churn. (XGBoost, Random Forest, etc.) and optimize (parameter tuning) 
- Evaluation: Evaluate model performance with accuracy, precision, recall.
   - Is the model performance > 0.8 F1 score?
   - When we input specific user information, can we get the churn decision right away?


## Milestone 3: Trend Analysis
- Temporal Trend Identification: Identify trending content over time. 
- Cross-Platform Analytics: Integrate external ratings and popularity metrics. 
- Visualize top genres, movies, and content patterns.


## Milestone 4: Visualization Manager & Dashboard (Front-end)
- Marcus's View (Marketing): Build the "Risk Segment List" interface with dynamic churn probability sliders and CSV export functionality (Use Case 1).
- Sarah's View (Studio): Develop the "Quality Elasticity Chart" and "Title Performance Matrix" correlating ratings with retention ROI (Use Case 2).
- Puja’s View (Product): Create the "Feature Engagement Heatmap" to visualize drop-off rates across search and recommendation logs (Use Case 3)
- Interactive Filters: Implement interactive global filters for genre, plan type, and content category.


## Milestone 5: Integration & System Validation
- End to End testing: Conduct systematic validation of all three use cases to ensure system behavior aligns with defined Interaction Diagrams.
- Final Documentation: Finalize Functional/Component Specification and README. 
- Final Report: Summarize final business insights (e.g., high-ROI content identifying, friction points in UI) and prepare the project repository for deployment.