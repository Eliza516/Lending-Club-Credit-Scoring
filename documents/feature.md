# Preprocessing 1: 

## 1. Choose labels

```
df.loan_status.value_counts()Current               500937
Fully Paid            358629
Charged Off            99099
Late (31-120 days)     13203
In Grace Period         6337
Late (16-30 days)       3414
Default                   36
Name: loan_status, dtype: int64
```

We see that there are seven types of the loan status. However, in this tutorial, we are interested in two classes: 1) Fully paid: those who paid the loan with interests and 2) Charged off: those who could not pay and finally charged off. Therefore, we select the data sets for these two classes:

@[medium.com/data-science/financial-data-analysis-80ba39149126](https://medium.com/data-science/financial-data-analysis-80ba39149126)

@ Managing Dataset Shift by Adversarial Validation for Credit Scoring

* November 2022
* [Lecture Notes in Computer Science](https://www.researchgate.net/journal/Lecture-Notes-in-Computer-Science-1611-3349?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19)

DOI:[10.1007/978-3-031-20862-1_35](https://doi.org/10.1007/978-3-031-20862-1_35)

* In book: PRICAI 2022: Trends in Artificial Intelligence (pp.477-488)

## 2. Remove features

* Remove features associated with >85% missing values
* Remove constant features
* Remove duplicates features
* Remove duplicate rows
* Remove highly collinear features (In part 3 EDA)
