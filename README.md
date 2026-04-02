# acs income fairness
## Objective
Binary classification of Income > $50,000 or below.

Train & evaluate a model for fairness.  <br>
See how performance metrics can indicate a model might skew unfairly.  <br>
Apply model remediation techniques to minimise the differe in error rates b/w demographic groups.

## Technologies
- TensorFlow Model Analysis
- Fairness Indicators
- TensorFlow Model Remediation
  - MinDiff
- TensorFlow (keras)

## Dataset
ACS (American Community Survey) Income Dataset
- it's an updated & reconstructed version of the UCI Adult dataset
- it's one of several datasets created by Ding et al - an alternative to the UCI Adult dataset.
- contains 1,664,500 datapoints
- from USA-wide (50 states + Puerto Rico) ACS Public Use Microdata Sample (PUMS), 2018.
- includes individuals
  - > 16 years of age
  - worked ≥ 1 hour / week in the past year
  - income ≥ $100

### Features include
- Education
- Employment
- Marital Status

### Label
Income (PINCP in Dataset)

We convert the Target value to Binary for this purpose:
- 1 for > 50000.0
- 0 for other

### Sensitive Attributes in the dataset include:
- Age
- Race
- Sex  <br>
We use these to help us figure out the fairness of the model across demographics.

## Project Workflow
1. Import Dataset
2. Change Target value to Binary
3. Define & Configure Base Model
4. Convert to `tf.data.Dataset`
   - 'cause MinDiff requirement
5. Training Set split (80% of original dataset)
6. Train & Evaluate Base Model
7. Evaluate tested Base Model for Fairness
   - across demographic groups
8. Explore results w/ different metrics to identify fairness concerns
   - overall performance vs demographic group

### The Model
We use a simple Keras model.
Minimally tuned and with Functional API for preprocessing.

### Base Model Fairness Evaluation
  - chosen sensitive attribute = `SEX` (biological sex and not gender)
  - we use Fairness Indicators to compute metrics and visualise results here
  - we use `model.predict(test_set)` to generate o/p predictions to be stored in a column on   the dataframe.
  - we create `eval_config` to use Fairness Indicators & it includes:
    - prediction column name
    - target label column name
    - list of metrics to compute
      - AUC
      - ExampleCount
      - BinaryAccuracy
      - ConfusionMatrixPlot
    - the Sensitive Attribute

## The Issues we found in Fairness


## Insights


#

So just for the record:
The model and prediction task here isn't really realistic.  <br>
Firstly, datasets don't often include sensitive attributes and it's never a good idea to impute those values.  <br>
And second, here we perform fairness using MinDiff in just one category. And that's:
1. 1 particular metric certainly doesn't ensure overall fairness.
2. And we are literally just minimising the differences.

P.S
Here we've used the sensitive attribute `SEX` for computing the fairness.  <br>
But without a separate gender expression category, this could possibly make room for ambiguity.  <br>
Ideally, you would have a separate gender column as well.

This is just an exercise in understanding how one could mitigate undesirable biases in a trained model.  <br>
It's about the approach and is very specific to just that:
- evaluating a trained model's performance
- & minimising the error rates b/w demographic groups

#

### References
For mapping codes of the values in the dataset, refer:
https://www.google.com/url?q=https%3A%2F%2Fproceedings.neurips.cc%2Fpaper_files%2Fpaper%2F2021%2Ffile%2F32e54441e6382a7fbacbbbaf3c450059-Supplemental.pdf
&
https://www.google.com/url?q=https%3A%2F%2Fwww2.census.gov%2Fprograms-surveys%2Facs%2Ftech_docs%2Fpums%2Fdata_dict%2FPUMS_Data_Dictionary_2018.pdf

### Citations
> Ding, Frances, Moritz Hardt, John Miller, and Ludwig Schmidt. "[Retiring adult: New datasets for fair machine learning.](https://proceedings.neurips.cc/paper_files/paper/2021/hash/32e54441e6382a7fbacbbbaf3c450059-Abstract.html)" Advances in neural information processing systems 34 (2021): 6478-6490.

> Sarah Flood, Miriam King, Renae Rodgers, Steven Ruggles, and J. Robert Warren (2020). Integrated Public Use Microdata Series, Current Population Survey: Version 8.0 [dataset]. Minneapolis, MN: IPUMS. https://doi.org/10.18128/D030.V8.0
