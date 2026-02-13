
### **Background** 
Build an automated pipeline that identifies high-risk subscribers using watch history, search logs and recommendation interaction.

### **User profiles** 
**Profile 1**: Product Manager (Puja)
**Role**: Feature Strategist
**Goal**: Ensure retention features are integrated into the UI
**Domain and computing knowlege**: Moderate
**Primary View**: Feature usage heatmap

**Profile 2**: Marketing Manager (Marcus)
**Role**: Campaign strategist
**Goal**: Design and deploy retention offers to the right customers
**Domain and computing knowlege**: Moderate
**Primary view**: Risk Segment list

**Profile 3**: Content Acquisition/Studio executive (Sarah)
**Role**: track ROI of the library
**Goal**: Renewing or cancelling TV shows
**Domain and computing knowlege**: Domain knowledge high and computing knowledge Low
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
**Use Case 1: Discount Simulation**
**Actor** : Marcus
**Objective**: Evaluate whether a 5% discount to high-risk subscribers generates a positive return
**Step 1: Identify high risk users**
Marcus navigates to churn prob. slider, filters users with P(churn) > 0.85
The dashboard dynamically recalculates key metrics and shows:
- Total users in the segment
- Average monthly subscription value
**Step 2: Understand why users might churn**
Marcus clicks on "Churn Drivers" chart which highlights top risk factors, examples: 
- No content watched in the last 10 days
- Declining weekly engagement trend
**Step 3: Simulate the discount impact**
Simulating Financial impact
Marcus selects "Simulate offer". He inputs 5% discount, duration= 3 months. The dashboard calculates total cost of discount, revenue retained under varying save-rate scenarios and break-even retention.
**Step 4:** 
Marcus clicks "Export segment". The dashboard pushes the filtered cohort into the Netflix email automation platform. 

**Use case 2: Optimizing content investment Using movie ratings**
Actor : Sarah, Studio executive
Goal : Decide whether investing in higher-rated content( 8+ IMDb) produces better long-term retention than increasing content volume.

Step 1: Identifying at risk viewers
Sarah opnens "Content and retention" tab.
She filters for:
- top 40% lifetime value users
- moderate churn prob. (0.5-0.7)
-Active in the last 60 days
The dashboard shows:
- $6M in quarterly revenue exposure
-65% watches content 7.8 + on IMDB
-Users who primarily watch content below 6.5 churn 2X faster
Step 2: Sarah switches to quality elasticity viz
The chart shows:
Average churn prob. by IMDb rating bucket
Completion rate by rating tier
Post-completion retention window.























