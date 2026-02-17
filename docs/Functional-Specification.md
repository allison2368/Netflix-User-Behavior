# Functional Specification

### **Background**
Build an automated pipeline that identifies high-risk subscribers using watch history, search logs and recommendation interaction.


### **User profiles**
**Profile 1**: Product Manager (Puja)
**Role**: Feature Strategist
**Goal**: Ensure retention features are integrated into the UI
**Domain and computing knowledge**: Moderate
**Primary View**: Feature usage heatmap


**Profile 2**: Marketing Manager (Marcus)
**Role**: Campaign strategist
**Goal**: Design and deploy retention offers to the right customers
**Domain and computing knowledge**: Technically proficient in using BI tools (Power BI, Tableau). Can interact with complex filters and export CSVs but cannot write SQL or Python code.
**Primary view**: Risk Segment list


**Profile 3**: Content Acquisition/Studio executive (Sarah)
**Role**: track ROI of the library
**Goal**: Renewing or cancelling TV shows
**Domain and computing knowledge**: High domain expertise in content ROI but minimal computing knowledge. Requires a simplified, high-level graphical interface with one-click insights.
**Primary View**: Title performance matrix


### Data Sources
1. Netflix 2025: User Behavior dataset
This is a synthetic dataset with 210K+ records.
**Dataset structure**
It has 6 interconnected tables.
|**Files**|**Records**|**Description**|
|**users.csv**|10300|Demographics+Subscription|
|**movies.csv**|1040|Content metadata+ratings|
|**watch_history.csv**|105000|viewing sessions and behavior|
|**recommendation_logs.csv**|52000|Algorithmic recommendations|
|**search_logs.csv**|26500|User search queries|
|**reviews.csv**|15450|Text reviews+sentiment|


2. Movie/show ratings from TMDb API


### Use Cases
**Use Case 1: Identify and Export High-Risk Churners for Targeted Discounts**
**Actor** : Marcus, Marketing Manager
**Objective**: Identify users with >85% churn probability and export them for a discount campaign.
**Step 1: User - Marcus sets the “Churn Probability” slider threshold to > 85% on the dashboard UI.**
**Step 2: System (Data Manager) - Queries the preprocessed watch_history and users tables to filter IDs meeting the threshold. **
**Step 3: System(Analytics & Prediction Engine) - Calculates the “Total Revenue at Risk” based on the filtered segment’s monthly subscription fees.**
**Step 4: User - clicks the “Export Segment” button**
**Step 5: System (Visualization & Interaction Manager) - Generates a CSV file containing the filtered user list, ready for marketing automation integration.**




**Use case 2: Optimizing content investment Using movie ratings**
**Actor**: Sarah, Studio executive
**Objective** : Compare the retention impact of high-rated content versus content volume to optimize budget allocation.
**Step 1: User - Sarah filters the dashboard for users in the top 20% lifetime value and active in the last 60 days.**
**Step 2: System (Data Manager) - Joins `users.csv` and `watch_history.csv` to isolate the cohort and calculates “Revenue Exposure”.**
**Step 3: System (Visualization & Interaction Manager) - Renders the "Quality Elasticity Chart” by correlating IMDB ratings (in movies.csv) with user completion rates. **
**Step 4: User - Sarah clicks on a specific rating bucket (e.g., 8.0+) to see which title drives the longest retention window. **
**Step 5: System (Visualization & Interaction Manager) - Displays a detailed performance matrix of content metadata and user sentiment scores from reviews.csv. **




**Use case 3: Feature Engagement vs. Churn Correlation Analysis**
**Actor** : Puja (Product Manager)
**Objective** : Identify underperforming platform features causing user frustration.
**Step 1: User - Puja selects the “Feature Engagement” tab and filters by “Search Logs”.**
**Step 2: System (Data Manager) - joins search_logs.csv and recommendation_logs.csv to calculate the “Null Search Rate” (searches with no clicks)**
**Step 3: System (Visualization & Interaction Manager) - displays a heatmap of features with the highest drop-off rates**
**Step 4: User - Puja hover over a specific hotspot (high drop-off area) on the heatmap. **
**Step 5: System (Visualization & Interaction Manager) - Displays a tooltip showing the top 5 failed search queries associated with that feature.** 
