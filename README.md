![Banner](assets\banner.gif)
---

# Sentiment Analysis Project

**Project Overview:**
The Sentiment Analysis Project is designed to automatically detect and classify sentiment in textual data as positive or negative. The project leverages **Natural Language Processing (NLP)** techniques along with **deep learning models**, focusing on robust preprocessing, advanced text representations, and an interactive frontend for real-time prediction. The project emphasizes experimentation with multiple embedding techniques and model architectures to achieve high accuracy and generalization.

---

### **Data Preprocessing**

Effective preprocessing is critical for accurate sentiment prediction. The following steps were implemented:

* **Noise Removal:** URLs, mentions, hashtags, emojis, punctuation, special characters, numbers, and diacritics were removed or normalized.
* **Text Normalization:** Lowercasing and lemmatization were applied to reduce variability in word forms.
* **Tokenization:** Text was split into meaningful tokens to prepare for vectorization.
* **Contraction Expansion:** Words like "can't" or "won't" were expanded to "cannot" and "will not" for semantic clarity.
* **Custom Cleaning Pipeline:** Implemented a reusable pipeline using a custom sklearn Transformer for consistent preprocessing across training and prediction.
    
    Note: I haven't applied stop word in the preprocessing step becuase stopwords also help to adding context and structure to the text.

---

### **Advanced Text Representation**

To capture the semantic meaning of text effectively, multiple embedding and representation strategies were explored and evaluated. Initially, **pretrained word embeddings** were tested for their ability to encode contextual information learned from large corpora. Among these, **GloVe Twitter embeddings** proved particularly valuable, as they were trained on extensive social media data, enabling the model to understand slang, abbreviations, and informal expressions typical of online text. Other pretrained options, such as **FastText**, were also experimented with because of their ability to capture subword information, which helps in handling rare words, misspellings, or domain-specific variations in vocabulary.

Beyond pretrained embeddings, **traditional embedding approaches** were considered. The **Continuous Bag-of-Words (CBOW)** model was applied to understand context by predicting words based on surrounding terms, while **n-gram representations** were evaluated to capture localized sequences and patterns of words that convey sentiment. These methods provided insights into how sequence information and local dependencies could enhance sentiment detection.

Additionally, **custom embeddings** were trained on the project-specific dataset to adapt to domain-specific sentiment patterns. These embeddings were particularly useful for understanding expressions unique to the dataset that might not be well-represented in general-purpose pretrained embeddings. The performance of these custom-trained embeddings was compared against pretrained options to identify the most effective approach.

Throughout experimentation, embeddings of varying dimensions—**25, 50, 100, 200, and 300**—were tested to evaluate the trade-off between computational complexity and representational power. While lower-dimensional embeddings offered faster training, they struggled to capture sufficient semantic nuance. In contrast, **200-dimensional embeddings consistently delivered the best performance**, balancing richness of representation with generalization to unseen data. Ultimately, the project adopted **GloVe Twitter 200d embeddings** for the final model, leveraging their semantic richness and robustness for accurate sentiment prediction across diverse text inputs.

---

### **Model Architecture**

The project utilizes a deep learning **LSTM (Long Short-Term Memory)** network, which is specifically designed to handle sequential data and capture dependencies across time steps in text. At its core, the LSTM layers are responsible for learning long-term relationships between words in a sentence, allowing the model to understand context and subtle sentiment cues that span across multiple words or phrases.

Following the sequential layers, dense layers are incorporated to capture more complex, non-linear relationships in the transformed features. These fully connected layers employ activation functions such as ReLU for Dense layer and Tanh for LSTM to introduce non-linearity, enabling the model to learn intricate patterns in sentiment expressions that simple linear transformations cannot capture.

To ensure stable training and improve generalization, he_normal weight initialization is used for Dense layers, the network also incorporates normalization and regularization techniques. BatchNormalization (For dense layer) and LayerNormalization (For LSTM layer) help maintain consistent feature distributions across layers, speeding up convergence and reducing internal covariate shift. Dropout layers are applied strategically to randomly deactivate neurons during training, preventing the network from overfitting to the training data and promoting robustness in predictions for unseen text.

This combination of sequential LSTM layers, dense non-linear transformations, and normalization with dropout ensures that the model effectively learns semantic patterns in text while remaining generalizable, robust, and accurate for sentiment prediction tasks.

---

### **Evaluation**

The trained model was thoroughly evaluated on unseen test data and hold out validation data to measure its predictive performance:

* **Standard Metrics:** The following metrics were calculated using the best threshold for classification:

  * **Accuracy:** 0.9838 → Indicates that ~98% of predictions are correct overall.
  * **Precision:** 0.9903 → Very few false positives; model is highly confident when predicting positive sentiment.
  * **Recall:** 0.9773 → Most positive sentiments are correctly identified, with few missed instances.
  * **F1-score:** 0.9838 → Excellent balance between precision and recall, showing robust performance.
  * **ROC AUC:** 0.9987 → Model is near-perfect in ranking positive instances higher than negatives.
  * **Loss:** 0.0468 → Very low binary cross-entropy loss, indicating the model is well-optimized.
---

### Preview
![preview](assets\project_preview.png)