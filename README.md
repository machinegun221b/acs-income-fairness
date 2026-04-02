# ACS Income Fairness
## Objective
- Train & evaluate a model for fairness.
- See how performance metrics can indicate a model might skew unfairly.
- Apply model remediation techniques to minimise the differe in error rates b/w demographic groups.

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
  - age > 16 years
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

(Binary classification of Income > $50,000 or below.)

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
9. Minimise differences across demographic groups with MinDiff
   -  Create Positively-Labelled Subsets to setup TF model remediation
   -  Pack original dataset + sensitive + non-sensitive subsets
10. Train & Evaluate MinDiff Model
11. Remediated Fairness Evaluation 

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
- the FNRs (false negative rates) reveal disproportionate outcomes b/w the Female & Male groups.
  - equality of opportunity
- model training goal is to decrease the FNR gap between the groups.
  - TensorFlow Model Remediation → during training to intervene &
    - minimise error rate b/w the groups

But to do so, we need to
### Setup Positively-Labeled Subsets
1. Sensitive Group/Protected Class
   - only +ve labelled female examples
2. Non-Sensitive Group (Any group that's not Protected Class basically)
   - only +ve labelled male examples
  
## Minimise the differences with MinDiff
We equalise the distributions between the subsets using MinDiff:
- a TF model remediation technique
- helps balance error rates (FNR here) b/w demographic groups
- works by penalising distributional differences during training

### again, we need
`tf.data.Dataser` to pass as i/p into `MinDiffModel`  <br>
we use the same helper function as with the base model:  <br>
`dataframe_to_dataset`  <br>
to convert subsets into `tf.data.Dataset`

### Pack Datasets for MinDiff Model
do this only after ensuring all datasets are batched before packing together, which are the:
- original Dataset
- sensitive subset
- non-sensitive subset

These are packed into a single dataset with:  <br>
`min_diff.keras.utils.pack_min_diff_data()`

This completes ALL data prepping for this project flow so far.

## Train MinDiff model
- wrap base model in MinDiffModel and compile
  - we're using the same configuration
- pass in packed dataset for training

## MinDiff Model Evaluation + Remediated Fairness Evaluation 
Same evaluation process as base model. To reiterate:
- we generate predictions using MinDiff model on test set
- append predictions as column on dataframe
- pass colmn onto Fairness Indicators for evaluation

## Insights
In the initial model - Base Model, we see in the Fairness Indicators exploration:
- Metric that performed equally well across demographic groups: 
The overall AUC for the base model was around 0.88, with male and female groups performing just as well with 0.87 and 0.88, respectfully. A performance metric like the AUC would lead one to believe that the model performs well across groups.
- Metric that was disproportionate across demographic groups - even though its overall performance seemed promising: 
However, when evaluting with respect to the false negative rate, the results show that performance is disappropriately favoring males, with female performance is nearly 27% worse than overall baseline. In fact, what the graphs reveal is that males perform better than the baseline by around 16%.

And in the final remediated MinDiff model, we see:
- AUC indicates that the MinDiff model performance, compared to the base model, as a result of penalising the model during training for differences in error rates, 
Though applying MinDiff may come with some performance tradeoffs in comparison to the original task, in this exercise, the MinDiff model performed nearly equally as well as the base model, at least in terms of AUC. What this is suggesting is that, in this context, MinDiff can be effective while not worsening overall performance.
- FNR reveals that between the MinDiff and base model - the differences are 
Here is where we see the MinDiff performing better than the base model. Not only is the gap in error rates between male and female smaller, but the FNR for female went down drastically from nearly 35% all the way to 30%, while the overall performance still remains relatively the same (29% in MinDiff vs. 28%)

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
- [ACS Supplemental](https://www.google.com/url?q=https%3A%2F%2Fproceedings.neurips.cc%2Fpaper_files%2Fpaper%2F2021%2Ffile%2F32e54441e6382a7fbacbbbaf3c450059-Supplemental.pdf)
- [2018 ACS PUMS DATA DICTIONARY](https://www.google.com/url?q=https%3A%2F%2Fwww2.census.gov%2Fprograms-surveys%2Facs%2Ftech_docs%2Fpums%2Fdata_dict%2FPUMS_Data_Dictionary_2018.pdf)

### Citations
> Ding, Frances, Moritz Hardt, John Miller, and Ludwig Schmidt. "[Retiring adult: New datasets for fair machine learning.](https://proceedings.neurips.cc/paper_files/paper/2021/hash/32e54441e6382a7fbacbbbaf3c450059-Abstract.html)" Advances in neural information processing systems 34 (2021): 6478-6490.

> Sarah Flood, Miriam King, Renae Rodgers, Steven Ruggles, and J. Robert Warren (2020). Integrated Public Use Microdata Series, Current Population Survey: Version 8.0 [dataset]. Minneapolis, MN: IPUMS. https://doi.org/10.18128/D030.V8.0
