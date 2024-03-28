# GroupCE

Code to implement the optimization models described in the Numerical Illustration section in prepint *One-for-many Counterfactual Explanations by Column Generation* by Andrea Lodi and Jasone Ramírez-Ayerbe. 
The prepint can be found here: https://arxiv.org/abs/2402.09473

### Requirements

To run the model, the gurobi solver is required. Free academics licenses are available. 


### Files

* 'column_gen_counterf_gurobipy_heuristics.py': column generation framework
* 'globalcounterf_sparsity_nn.py': load the dataset, create the classification model, solve the MIP model (to compare it with the CG) and run the CG framework
