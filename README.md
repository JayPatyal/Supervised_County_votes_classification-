## Supervised Learning

Using the provided dataset, implementing 3 different classification algorithms (k-nearest neighbor, perceptron, and decision tree) to determine whether a candidate gets the majority of votes. 

## Data

The attached csv file contains all the data. The run file handles importing it and converting it to numpy arrays. A description of the dataset is in the run file.

## Algorithms

### K-Nearest Neighbor

Using Euclidean distance between the features. Choosing a k value and use majority voting to determine the class. The k value is provided to the knn class. 

### Perceptron

For the perceptron, multiply the inputs by a weight matrix and then pass the output through a single heaviside function (a.k.a. step function) to get the output.

### Decision Tree

ID3 stands for Iterative Dichotomiser 3 and is named such because the algorithm iteratively (repeatedly) dichotomizes(divides) features into two or more groups at each step.

#==========================================================Data==========================================================
# Number of Instances:	
653
# Number of Attributes:
35 numeric, predictive attributes and the class

# Attribute Information:

We have 35 variables for 653 counties, including demographics, covid info, previous election 
results, work related information.
percentage16_Donald_Trump	
percentage16_Hillary_Clinton	
total_votes20	
latitude	
longitude	
Covid Cases/Pop	
Covid Deads/Cases	
TotalPop	
Women/Men
Hispanic
White	
Black	
Native	
Asian	
Pacific	
VotingAgeCitizen	
Income	
ChildPoverty	
Professional	
Service	
Office	
Construction	
Production	
Drive	
Carpool	
Transit	
Walk	
OtherTransp	
WorkAtHome	
MeanCommute	
Employed	
PrivateWork	
SelfEmployed	
FamilyWork	
Unemployment


# Class Distribution:
328 - Candidate A (1), 325 - Candidate B (0)
