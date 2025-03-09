# Text Classification in Spanish using BERT
## Project Overview

This project implements a **text classification system using BERT** (Bidirectional Encoder Representations from Transformers) to **categorize Spanish messages into three categories**:
1. Information Requests
2. Complaints
3. Recommendations

The goal is to demonstrate the potential of pre-trained language models for Spanish text classification, covering data preparation, model training, evaluation, and error analysis.

## Key Features

- Utilizes the pre-trained model:(dccuchile/bert-base-spanish-wwm-uncased).
- Implements a three-class classification system.
- Applies data augmentation techniques.
- Provides detailed evaluation of model performance.
- Explores challenging examples to test the model’s limits.

## Why It Matters

**Efficient classification of user messages is essential for improving customer service, detecting key trends in feedback, and prioritizing responses.**
This project shows how state-of-the-art language models can be leveraged to understand customer input in Spanish, a language that is often underrepresented in NLP research.

## Dataset

- Dataset is embedded directly in the notebook.
- Consists of Spanish messages labeled into the three categories: requests, complaints, recommendations.
- Augmented to ensure 100 examples per class to improve model training.

## Technologies Used

- **Python**
- **Jupyter Notebook**
- **Pandas** for data manipulation
- **Scikit-learn** for data splitting and evaluation metrics
- **Simpletransformers** for easy implementation of BERT models
- **Matplotlib y Seaborn** Matplotlib & Seaborn: for result visualization

## Installation and Usage

### Option 1: Local Execution

1. Clone the repository:
   git clone https://github.com/BeaEsparcia/Spanish_Text_Classification_BERT.git
cd clasificacion_texto_espanol_bert
2. Install the required packages:
   pip install -r requirements.txt
3. Launch Jupyter Notebook:
   jupyter notebook
4. Open Clasificacion_texto_espanol_BERT.ipynb and run all cells

### Option 2: Run on Google Colab

1. Open Google Colab.
2. Upload Clasificacion_texto_espanol_BERT.ipynb.
3. Ensure runtime is set to GPU:
   - Go to "Runtime" > "Change runtime type"
   - Select "GPU"
4. Run all cells.

- If needed, ad:
  !pip install simpletransformers pandas scikit-learn matplotlib seaborn

## Methodology

### Preprocessing & Data Augmentation

- Conversion to lowercase.
- Handling ambiguous messages.
- Augmenting the dataset to 100 examples per class.

### Model Selection

- Initial tests with an English BERT model led to poor results.
- Switching to dccuchile/bert-base-spanish-wwm-uncased improved classification.

### Training Process

- Used Simpletransformers for efficient fine-tuning.
- Applied cross-validation and monitored overfitting signs.

### Evaluation

- Precision, Recall, and F1-score calculated for each class.
- Error analysis performed on misclassified messages.

## Results & Observations

- Overall Accuracy: Improved after switching to a Spanish pre-trained model.
- Performance on challenging cases: 47% accuracy on specially curated hard-to-classify messages.
- Key challenges:
     - Ambiguity between categories (some requests felt like recommendations).
     - Mixed context within a single message.
     - Linguistic complexity (informal language, sarcasm, abbreviations).

## Limitations & Future Improvements

- **Expand dataset:** Collect more diverse and complex messages.
- **Fine-tuning:** Explore hyperparameter tuning and regularization.
- **Advanced preprocessing:** Explore techniques like entity recognition or semantic clustering.
- **Compare with other architectures:** Test DistilBERT, RoBERTa, or newer multilingual models.

## What I Learned

- **Model choice matters:** Starting with an English model gave poor results, but switching to a Spanish-specific model was a game changer.
- **Data quality is everything:** Adding just a few challenging examples exposed weaknesses that traditional metrics didn’t reveal.
- **Spanish NLP still needs work:** Many models and tools are heavily optimized for English, making it crucial to test and adapt when working in Spanish.

## Contributions

Contributions are welcome! Please open an issue to propose changes or improvements.

## License

This project is licensed under the MIT License. See the LICENSE.md file for details.

## Contact

Beatriz Esparcia - esparcia.beatriz@gmail.com
LinkedIn: www.linkedin.com/in/beaesparcia



   





