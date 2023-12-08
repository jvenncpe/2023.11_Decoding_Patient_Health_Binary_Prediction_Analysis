#### *Capstone Project: November 2023*

<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/bf1829e0-91a2-48a3-a589-a53d518e539a")
"/></p>

# Decoding Patient Health: Binary Prediction Analysis

Capstone project output from "SP801 Statistical Analysis and Modeling Using Excel" facilitated by Project SPARTA from Development Academy of the Philippines. The course aims to equip participants with practical skills in data science and machine learning using Python libraries. Learners explore algorithm implementation and identify the most suitable analytical techniques for specific business needs in the capstone project output of 'SP901 Data Science and Machine Learning Using Python.

## Overall Peer Grade Assessment

<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/bf1829e0-91a2-48a3-a589-a53d518e539a")
"/></p>

## Activity

<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/22f1b954-0834-4f14-b7cd-735e67bc9dfd")
"/></p>

## Dataset Context

<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/bf1829e0-91a2-48a3-a589-a53d518e539a")
"/></p>

Each row in the dataset represents a unique patient with different attributes recordeds, potentially in the context of medical diagnostics. The "Failure.binary" column indicates a binary classification for immediate medical attention. 

## Criteria

<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/9b39fcd7-9799-44f6-bf4c-8608a5892c1a")
"/></p>

## Peer Grade Assessment Breakdown
<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/568672f0-0d2b-4bb9-904c-334b8834cfeb")"/>
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/2037e97e-8b64-429a-91fd-8c7509ceb562")"/>
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/1cf06b0d-7ab5-49ec-8c61-e29c9dbfeae3")"/>
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/cedbc3fb-2d31-4406-a2c4-36bf1c051565")"/>
</p>


---
## Methodology and Implementation
### 1. Perform Exploratory Analysis (Scaling, PCA, Unbalanced)

<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/58830b1c-3f8b-440f-a32e-4a255e718d27")
"/></p>

- #### Preprocessing - Loading of Dataset and Exploratory Data Analysis (EDA):
  - Loading of Dataset:
    - Involves importing the dataset into a dataframe, laying the groundwork for further analysis.
  - Exploratory Data Analysis (EDA):
    - A crucial step for a comprehensive understanding of the dataset.
    - Tasks involve reviewing the dataset's shape, inspecting initial and final rows, displaying column names, assessing data types, and identifying null values using methods like .info(), .isnull() and etc., such as below:

<table align="center">
  <tr>
    <td width="33%" align="center">Knowing the shape (.shape)</td>
    <td width="33%" align="center">First few rows (.head)</td>
    <td width="33%" align="center">Last few rows (.tail)</td>
  </tr>
  <tr>
    <td width="33%" align="center">Column names (.columns)</td>
    <td width="33%" align="center">Summary of the dataset (.info())</td>
    <td width="33%" align="center">Checking of missing values (.isnull())</td>
  </tr>
  <tr>
    <td width="33%" align="center">Dataset distribution (.describe() & .value_counts())</td>
    <td width="33%" align="center">Correlation of the columns (.corr())</td>
    <td width="33%" align="center"></td>
  </tr>
</table>

<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/19d5b772-227e-442b-8af8-75242d0f4f06")
"/></p>

- #### Preprocessing - Scaling:
  - The primary goal is to normalize features to a uniform scale.
  - We perform data splitting and "early column dropping" to enforce a clear separation between the training and testing datasets to prevent any inadvertent mixing of the two sets during subsequent preprocessing.
  - MinMaxScaler from the sklearn.preprocessing module is utilized to apply consistent scaling across features within the training and test datasets (X_train and X_test).

<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/98206479-7450-4c5e-8f3a-b10be0e93d3d")
"/></p>

- #### Preprocessing - Principal Component Analysis (PCA):
  - PCA, or Principal Component Analysis, serves the purpose of reducing dimensionality within the dataset, aiding in data compression while preserving vital information.
  - The determination of the number of components to retain (n_components) relies on the explained variance percentage.
  - Using the PCA module from sklearn.decomposition, the scaled training and test datasets are subjected to fitting and transformation for effective dimensionality reduction.

<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/18fff10a-6871-4ef0-8a47-c1c7aa822cd6")
"/></p>

- #### Preprocessing - Handling Unbalanced Data:
  - This step focuses on addressing class distribution within the training dataset, particularly essential in unbalanced datasets where one class significantly outweighs the other.
  - Utilizing Matplotlib, a pie chart visualizes the distribution of binary classes, offering a clear representation of the dataset's class proportions.
  - Since the Failure Binary Distribution is unbalanced, we made use of SMOTE to have authentic samples added in the training dataset --- making it balanced.

<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/58830b1c-3f8b-440f-a32e-4a255e718d27")
"/></p>

### 2. Split the Data (Train, Validation, Test)
  - #### Preprocessing - Splitting into Training, Validation and Testing Sets:
    - Using train_test_split function, the dataset is divided into training and testing sets.
    - Further train_test_split on the resulting training set to obtain the validation dataset.
    - The proportions are distributed as follows: Train Size = 0.6, Validation = 0.15, Test Size = 0.25.
    - This allocation ensures a balanced distribution across training, validation, and testing, enhancing robust model evaluation and performance assessment.

### 3. Perform 10-Fold Cross-Validation and Grid Search

<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/3c5987c9-2280-44e1-92fc-a8790f4a54ce")
"/></p>

  - #### 10-fold Cross-Validation:
    - This method evaluates how effectively the model generalizes to new data by dividing the dataset into ten parts. Nine parts are used for training, and one part is for validation, iterating this process ten times. It provides a comprehensive assessment of the model's performance.

<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/9afd53f3-0e1c-4262-851c-b24e68a988d4")
"/></p>

  - #### Grid Search:
    -  A technique that systematically explores diverse parameter combinations within a predefined grid to identify the most optimal parameters for the machine learning model. This process aims to enhance the model's performance by finding the best configuration.
   
### 4. Compare the different classification medthods (Logistic Regressin, KNN, SVM, RF, XGBOOST)
  > I have a total of six (6) different classification models due to XGBOOST as I am not sure (from the instructions of the project capstone) if it was referring to the standard gradiant boosting or to the extreme gradiant boosting since based on the course ipnyb handouts, I have not found an extreme gradiant boosting --- only standard gradiant boosting.

  - #### Model Creation and Training:
    - The classification model is constructed using the training data, and the model is trained on this dataset.

  - #### Accuracy Score Computation: :
    - Calculate the accuracy scores separately for both the training and test datasets to gauge the model's performance on seen and unseen data.
  
<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/04353e1e-938c-4053-87ff-36f7c2d391e7")"/>
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/5025bd17-8afe-409b-b947-f9e1e7eac062")"/>
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/c6034806-855d-488b-bcbc-676d135bb281")"/>
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/47b9815d-ffe0-4cd1-b815-ed5b95b59782")"/>
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/9b9cb7d4-91c3-44a4-9005-e9440197de73")"/>
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/9dd4054d-3e3d-40fc-bde4-da10f08f0c29")"/>
</p>

  - #### Visualization for Comparison:
    - A column chart can be generated to visually compare and depict the accuracy scores obtained from the training and test sets, providing a clear understanding of the model's performance on both datasets.

### 5. Show evaluation metrics (ROC-AUC, accuracy, f-1 score)

<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/58830b1c-3f8b-440f-a32e-4a255e718d27")
"/></p>

  - #### Model Prediction:
    - Utilize the trained model to predict outcomes for the validation or test dataset.

<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/58830b1c-3f8b-440f-a32e-4a255e718d27")
"/></p>

  - #### Accuracy, ROC-AUC, and F-1 Score Calculation:
    - Determine the accuracy score, measuring the proportion of correctly classified instances.
    - Evaluate the Receiver Operating Characteristic - Area Under Curve (ROC-AUC), highlighting the model's performance across diverse thresholds.
    - Calculate the F-1 score, providing a balanced view between precision and recall and showcasing the model's accuracy in handling imbalanced classes.

<p align="center">
<img src="https://github.com/jvenncpe/2023.11-Decoding-Patient-Health-Binary-Prediction-Analysis/assets/35190918/58830b1c-3f8b-440f-a32e-4a255e718d27")
"/></p>

  - #### Visualization of Evaluation Metrics:
    - Use a column chart to visually compare the evaluation metrics such as accuracy, ROC-AUC, and F-1 score.
    - Each metric can be represented as a bar in the chart for easy comparison and understanding of their relative values.

## Results and Discussion

### Sales Growth Over the Years

![image](https://github.com/jvenncpe/Sales-Performance-Review/assets/35190918/64b44a16-5090-420a-bee8-05414a83263d)

The sales recovery growth from 2021 to 2022 was 1.23%, resulting in an increase of ₱49,532.00. Over the broader period, from 2018 to 2022, the average yearly sales growth was 0.70%, depicting an additional ₱28,220 of sales increase annually.

### Conversion Rate Over the Years

![image](https://github.com/jvenncpe/Sales-Performance-Review/assets/35190918/4d670707-f875-4f35-abd2-ff0cafeebe28)


So, the conversion rate from 2018 to 2022 stood at 9.46%, resulting in an increase of 1,529 total orders since inception. The average conversion growth rate during this period was 2.07%, depicting an additional 334 orders per year on average.

### Customer Value Segmentation

![image](https://github.com/jvenncpe/Sales-Performance-Review/assets/35190918/feffbb5e-6e18-4744-8428-91e3184d978c)


The top customer segment, constituting 17% of the total customer value segmentation, showcases higher average monetary value (₱450.90) and frequency (14 orders), with the latest purchase recorded on December 31, 2022.

Contrastingly, the immediate attention segment, occupying 21% of the total customer value segmentation, portrays a lower monetary average (₱338.95) and frequency (8 orders), with the latest purchase recorded on December 13, 2022, indicating potential areas needing prompt attention or improvement.

### Sales Representative Value Segmentation

![image](https://github.com/jvenncpe/Sales-Performance-Review/assets/35190918/dcb9649d-bfac-470e-bd66-161b29043b0a)


The outstanding sales representative segment, comprising 8% of the total sales representative segmentation pie chart, embodies exceptional attributes:

- "Need Attrition" Accounts: 2
- "Need Attention" Accounts: 2
- "At Risk" Accounts: 2
- "Loyal" Accounts: 6
- "Top" Accounts: 3


Meanwhile, the immediate attention sales representative segment portrays concerning aspects:

- "Need Attention" Accounts: 3
- "At Risk" Accounts: 5
- "Loyal" Accounts: 3
- "Top" Accounts: 2

This segment indicates a higher count of accounts needing attention and at risk, reflecting potential challenges or issues requiring immediate resolution or improvement.


---
![image](https://github.com/jvenncpe/Sales-Performance-Review/assets/35190918/1a3cc5c5-af01-4261-a652-4e1fc1b92375)

### *Overall, this project provided a comprehensive framework for analyzing QR code marketing data, empowering users to glean actionable insights, make informed decisions, and refine marketing strategies.*

> Youtube [Link](https://youtu.be/G5EHkQ5p3M8) for the full run of the story telling.



# Thank you!

