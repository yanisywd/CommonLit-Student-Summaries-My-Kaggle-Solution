# Student Summary Evaluator

A machine learning solution leveraging BERT to automatically assess and score student-written summaries. This system is designed to aid educators and online learning platforms in providing instantaneous feedback, enhancing learning outcomes.

## Overview

The code in this repository focuses on training a model to evaluate the quality of summaries written by students in grades 3-12. Our aim is to assess:

- **Representation of the main idea and details** from the source text.
- **Clarity, precision, and fluency** of the language used in the summary.

This system is critical as summary writing promotes critical thinking and is an effective method to improve writing abilities.

## Code Workflow

1. **Data Loading and Preprocessing**:
   - Loaded the training datasets: `summaries_train.csv` and `prompts_train.csv`.
   - Conducted basic preprocessing: text lowercase conversion and punctuation removal.
   - Implemented lemmatization using `spaCy` for textual normalization.

2. **Tokenization**:
   - Utilized the pre-trained BERT tokenizer.
   - Encoded summaries, added special tokens, and managed text padding and attention masks.

3. **Data Wrangling**:
   - Created a custom dataset, `SummarizationDataset`, to hold tokenized inputs and associated labels.
   - Split data into training (90%) and validation (10%) sets.

4. **Model Training**:
   - Loaded a pre-trained BERT model for sequence classification.
   - Established a training loop for 3 epochs.
   - Trained separately for `content` and `wording` scores using the mean squared error (MSE) loss.

5. **Model Inference**:
   - Loaded the test dataset: `summaries_test.csv`.
   - Predicted both `content` and `wording` scores for test data.

6. **Submission**:
   - Created a `submission.csv` file that contains the evaluated scores for each student summary.

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- spaCy
- torch
- transformers

By leveraging the power of BERT and a meticulous preprocessing pipeline, this solution aims to set new standards in the automated evaluation of student summaries.
