# Aspect-Based Sentiment Analysis for Mobile Phone Reviews Using AI Techniques

(7PAM2002: Data Science Master’s Project)

This project aims to perform **aspect-based sentiment analysis** on customer reviews of mobile phones from Flipkart. The primary objective is to classify sentiments and extract insights using advanced AI techniques.

---

Owned by: 

Name: Vullinkala Ratan Srivasthava Reddy

Student ID: 22031426

UNIVERSITY OF HERTFORDSHIRE


## Dataset  
- **Source:** [Kaggle Flipkart Cell Phone Reviews](https://www.kaggle.com/datasets/nkitgupta/flipkart-cell-phone-reviews)  
- **Content:** Reviews and product details of various mobile phones from Flipkart.

---

## **Input and Output**

### **Input:**
- Customer reviews of mobile phones from Flipkart in textual format.
- Corresponding product metadata such as brand, model, and ratings.
- Pre-processed text after cleaning, tokenization, and normalization.

### **Output:**
- Predicted sentiment labels (Positive, Negative) for each aspect of the review.
- Visualization of sentiment trends using bar plots and word clouds.
- Performance metrics (Loss, Accuracy, F1 Score) for different models.
- Comparison of model effectiveness in sentiment classification.

---

## **Methodology**  

### **Data Processing**
1. **Cleaning:**  
   - Removed HTML tags, decoded emoticons, and handled special characters to ensure clean and standardized text data.

2. **Tokenization:**  
   - Split reviews into sentences and words using word tokenization, preparing the data for model training.

3. **Text Normalization:**  
   - Removed stop words and performed stemming and lemmatization to reduce words to their root form for better model learning.

4. **Balancing:**  
   - Applied antonym augmentation to balance the dataset, specifically targeting negative reviews to ensure a more representative training set.

### **Exploratory Data Analysis (EDA)**
- Analyzed review lengths, active reviewers, and common terms using bar plots and word clouds to understand the dataset better.
- Conducted target distribution and n-grams analysis for deeper insights into the nature of reviews and sentiments.

### **Modeling**
Three models were trained and evaluated for aspect-based sentiment analysis:

1. **LSTM (Long Short-Term Memory)**
2. **BiLSTM (Bidirectional LSTM)**
3. **Enhanced BiLSTM (Proposed)**

These models were trained using different hyperparameter configurations, as detailed below.

---

## **Hyperparameter Details**

### **Hyperparameters for LSTM, BiLSTM, and Enhanced BiLSTM:**

1. **LSTM_hparam1 (Default Hyperparameters)**
   - **Embedding Dimension:** 50
   - **LSTM Units:** 128
   - **Dropout Rate:** 0.3
   - **Learning Rate:** 0.001
   - **Epochs:** 30
   - **Batch Size:** 64

2. **LSTM_hparam2**
   - **Embedding Dimension:** 64
   - **LSTM Units:** 128
   - **Dropout Rate:** 0.4
   - **Learning Rate:** 0.0005
   - **Epochs:** 30
   - **Batch Size:** 64

3. **LSTM_hparam3**
   - **Embedding Dimension:** 128
   - **LSTM Units:** 128
   - **Dropout Rate:** 0.5
   - **Learning Rate:** 0.0001
   - **Epochs:** 30
   - **Batch Size:** 32

4. **BiLSTM_hparam1, BiLSTM_hparam2, BiLSTM_hparam3**  
   These hyperparameters mirror the LSTM model configurations but with bidirectional LSTM layers for enhanced learning of sequence patterns.

5. **Enhanced BiLSTM_hparam1, Enhanced BiLSTM_hparam2, Enhanced BiLSTM_hparam3**  
   Enhanced models include dropout and L2 regularization techniques to prevent overfitting and improve generalization.

---

## **Results**

| **Model**                     | **Accuracy (%)** | **Precision** | **Recall** | **F1 Score** | **ROC-AUC** |
|-------------------------------|------------------|---------------|------------|--------------|-------------|
| **LSTM (H_param1)**            | **91.0**        | 0.91          | 0.94       | 0.94         | 0.95        |
| **BiLSTM (H_param1)**          | 89.0            | 0.89          | 0.93       | 0.92         | 0.94        |
| **Enhanced BiLSTM (H_param1)** | 89.0            | 0.89          | 0.94       | 0.92         | 0.94        |
| **LSTM (H_param2)**            | **91.0**        | 0.91          | 0.94       | 0.94         | 0.95        |
| **BiLSTM (H_param2)**          | 89.0            | 0.89          | 0.93       | 0.92         | 0.94        |
| **Enhanced BiLSTM (H_param2)** | 90.0            | 0.90          | 0.93       | 0.93         | 0.94        |
| **LSTM (H_param3)**            | **91.0**        | 0.91          | 0.94       | 0.94         | 0.95        |
| **BiLSTM (H_param3)**          | 89.0            | 0.89          | 0.94       | 0.93         | 0.94        |
| **Enhanced BiLSTM (H_param3)** | 89.0            | 0.89          | 0.95       | 0.93         | 0.94        |

**Table 4.1: Model Performance Comparison**  
This table summarizes the performance of LSTM, BiLSTM, and Enhanced BiLSTM models across different hyperparameter configurations. **LSTM** consistently achieved the highest accuracy, precision, recall, and F1 scores, with an overall strong ROC-AUC of 0.95.

---

## **Key Features**
- **Antonym Augmentation:** Improved dataset balance by increasing negative reviews, contributing to better model robustness.
- **Dropout and Learning Rate Scheduler:** These techniques helped prevent overfitting and ensured stable model convergence during training.
- **Comparative Model Analysis:** The results from different models with varying hyperparameters show trade-offs between performance and computational efficiency.

---

## **Conclusion**
The project demonstrated the effective use of advanced AI techniques for aspect-based sentiment analysis on mobile phone reviews. **LSTM** outperformed other models in terms of overall performance across various metrics. However, **Enhanced BiLSTM** also showed promising results in terms of recall for the positive class. The findings suggest that **LSTM** is the best model for sentiment analysis in this case, and it can be further fine-tuned for even better performance. This approach not only aids in understanding customer sentiments but can also be leveraged for making data-driven decisions in the mobile phone e-commerce sector.
