# DS2010 Final Project

Analyzing online shopping data to determine what affects a customer’s tendency to make a purchase.

## Running

Poetry is used to manage dependencies. To install dependencies, run:

```bash
poetry install
```

To run the project, run:

```bash
poetry run python shop-data.py
```

The plots are saved in the `plots` directory.

## Example Output

Here's the output for out dataset using a logistic regression model, as of 3 December 2024:

![output graph](docs/static/feature_importance.svg)

Here's an example of the information printed to the terminal when the program is run:

```txt
Online Shopping Purchase Prediction Analysis
============================================

Dataset Overview:
Number of sessions: 12330

Feature Statistics:
       Administrative  Administrative_Duration  ...        Region   TrafficType
count    12330.000000             12330.000000  ...  12330.000000  12330.000000
mean         2.315166                80.818611  ...      3.147364      4.069586
std          3.321784               176.779107  ...      2.401591      4.025169
min          0.000000                 0.000000  ...      1.000000      1.000000
25%          0.000000                 0.000000  ...      1.000000      2.000000
50%          1.000000                 7.500000  ...      3.000000      2.000000
75%          4.000000                93.256250  ...      4.000000      4.000000
max         27.000000              3398.750000  ...      9.000000     20.000000

[8 rows x 14 columns]

No missing values found in the dataset.

Purchase Distribution:
Revenue
False      0.845255
True       0.154745
Name: proportion, dtype: float64

Visitor Type Analysis:
                   Total Sessions  Conversion Rate
VisitorType
New_Visitor                  1694         0.249115
Other                          85         0.188235
Returning_Visitor           10551         0.139323

Shopping Behavior Statistics:
Average pages per session: 5.89
Average time spent: 1310.04 seconds
Purchase rate: 15.47%

Seasonal Purchase Patterns:
  Month  Conversion Rate
7   Nov         0.253502
8   Oct         0.209472
9   Sep         0.191964
0   Aug         0.175520
3   Jul         0.152778
1   Dec         0.125072
6   May         0.108502
4  June         0.100694
5   Mar         0.100682
2   Feb         0.016304

Model Performance:
Training accuracy: 0.888
Testing accuracy: 0.873

Cross-validation scores:
Mean: 0.822 (+/- 0.236)

Confusion Matrix:
[[2008   47]
 [ 266  145]]

Classification Report:
              precision    recall  f1-score   support

       False       0.88      0.98      0.93      2055
        True       0.76      0.35      0.48       411

    accuracy                           0.87      2466
   macro avg       0.82      0.66      0.70      2466
weighted avg       0.86      0.87      0.85      2466


Recommendations for Retailers:
- Focus on optimizing PageValues: Impact score 1.527
- Focus on optimizing ExitRates: Impact score 0.817
- Focus on optimizing Month Nov: Impact score 0.325
- Focus on optimizing Month Feb: Impact score 0.198
- Focus on optimizing Month May: Impact score 0.134
```
