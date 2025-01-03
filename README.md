# Aspect-Based Sentiment Analysis for Mobile Phone Reviews Using AI Techniques

(7PAM2002-0901-2024: Data Science Master’s Project)

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

## Methodology  

### **Data Processing**
1. **Cleaning:** Removed HTML tags, decoded emoticons, and handled special characters.  
2. **Tokenization:** Sentence segmentation and word tokenization.  
3. **Text Normalization:** Stop word removal, stemming, and lemmatization.  
4. **Balancing:** Used antonym augmentation for balancing negative reviews.

### **Exploratory Analysis**
- Analyzed review lengths, active reviewers, and common terms via bar plots and word clouds.  
- Target distribution and n-grams analysis for insights.  

### **Modeling**
Three models were trained and evaluated:  
- **LSTM**  
- **Bidirectional LSTM (BiLSTM)**  
- **Enhanced BiLSTM** (Proposed)  

---

## Results  

| Model            | Loss  | Accuracy | F1 Score |  
|-------------------|-------|----------|----------|  
| **LSTM**         | 0.4049 | 88.25%   | 91.19%   |  
| **BiLSTM**       | 0.3745 | 87.49%   | 91.19%   |  
| **Enhanced BiLSTM** | 0.3772 | **88.76%**   | **92.37%**   |  

The **Enhanced BiLSTM** model achieved the best performance with the lowest loss, highest accuracy, and F1 score. This highlights its effectiveness for e-commerce sentiment analysis.

---

## Key Features  
- **Antonym Augmentation:** Improved balance and model robustness.  
- **Dropout and Learning Rate Scheduler:** Prevented overfitting.  
- **Comparative Analysis:** Identified trade-offs between models.


## Conclusion  
This project demonstrates that advanced techniques like Enhanced BiLSTM can significantly improve sentiment analysis, aiding technological growth in the e-commerce sector.
