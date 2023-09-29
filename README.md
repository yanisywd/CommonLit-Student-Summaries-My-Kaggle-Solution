# Student Summary Evaluator: CommonLit - Evaluate Student Summaries

This repository hosts a machine learning solution developed for the Kaggle competition: **CommonLit - Evaluate Student Summaries**. It's tailored to leverage BERT in automatically assessing and scoring student-written summaries, aiming to assist educators and online platforms in delivering real-time feedback.

## Overview

In the backdrop of the CommonLit competition, the system here trains a model to evaluate summaries written by students from grades 3 to 12. The primary objectives are:

- **Representation of the main idea and details** taken from the source text.
- **Clarity, precision, and fluency** in the student's language usage in the summary.

This is more than a code—it's a bridge to enhance the pedagogic process. Summary writing not only promotes critical thinking but also stands out as a formidable technique to amplify writing proficiencies.

## Code Workflow

1. **Data Loading and Preprocessing**:
   - Data from Kaggle's competition datasets: `summaries_train.csv` and `prompts_train.csv` are loaded.
   - Underwent standard preprocessing—like transforming text to lowercase and eliminating punctuations.
   - Deployed `spaCy` for lemmatization, ensuring textual uniformity.

2. **Tokenization**:
   - Harnessed the might of the pre-trained BERT tokenizer.
   - Summaries were encoded, special tokens integrated, and both padding & attention masks were managed.

3. **Data Wrangling**:
   - A custom dataset named `SummarizationDataset` was curated to store tokenized inputs and their respective labels.
   - The data split: 90% for training and 10% for validation.

4. **Model Training**:
   - A pre-trained BERT model for sequence classification was summoned.
   - A training loop spanning over 3 epochs was established.
   - Dual training processes were executed—one for `content` and the other for `wording` scores, both using the mean squared error (MSE) loss.

5. **Model Inference**:
   - The testing phase made use of the `summaries_test.csv` dataset.
   - Predictions were rolled out for both `content` and `wording` scores.

6. **Submission**:
   - The culmination saw the creation of a `submission.csv` file, capturing the computed scores for each student's summary.

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- spaCy
- torch
- transformers

This solution, crafted for the Kaggle's CommonLit competition, harnesses the BERT model in synergy with a detailed preprocessing strategy, aspiring to redefine the benchmarks in automated evaluation of student-generated summaries.

