

from time import time

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder
import numpy as np

import gurobipy as gp
from gurobipy import GRB
from gurobi_ml import add_predictor_constr




class CFMasterProblem():

    """The master problem for generating global counterfatual explanations. 

    Parameters
    ----------

    T_max: int
        maximum number of features changed


    Attributes
    ----------
    X_ : ndarray, shape (self.n_samples, self.n_features)
        The input passed during :meth:`generate_mp`. The inputs should only
        contain values in {0,1}.
   
    
    #counterf_vars_: list(int),
    #    Stores the indices of counterfactuals variables x_j generated so far.
    #xi_vars_ : list(int),
    #    Stores the indices of positive penalty variables xi.
    
    #    Index of constraint (1b).
    counterf_dict_ : dictionary, (int->list)
        Dictionary of counterfactuals generated so far. The keys are the indices of the
        corresponding variables and the values are the lists containing the
        value of features that make the clause.
    generated_ : boolean,
        True when the master problem model has been generated. False otherwise.
    """

    def __init__(self,Tmax):
     

        
        self.generated_ = False
        
        self.Tmax_ = Tmax
        
        self.rmp_objective=0
        # Vars
        self.counterf_vars_ = None
        

        self.counterf_dict_ = {}
        self.selected_dict_={}
        self.features_dict_={}
        self.assignment_dict_={}
        self.columns_added=[]

        self.model_master = gp.Model('masterLP')
        
    def init_sol(self,X, sets_T, clas_model):

        self.T_original=sets_T[0]
        self.T_corr=sets_T[1]
        self.T_c=sets_T[2]

        self.X_=X
        self.n_samples=self.X_.shape[0]
        self.n_features=self.X_.shape[1]
        self.n_counterf=self.n_samples


        m=gp.Model('initsol')

        x_l={}
        for k in range(self.n_samples):
            x_l[k]=m.addMVar(self.n_features, vtype=GRB.BINARY, name="counterf[%d]"%k)
            #x_l[k]={}
            #for f in range(self.n_features):
            #    x_l[k][f]=m.addVar(vtype=GRB.BINARY,name="counterf[%d][%d]"%(k,f))

        y_clas={}
        for k in range(self.n_samples):
            y_clas[k]=m.addMVar(2, lb=-gp.GRB.INFINITY, name="y[%d]"%k)

        xi={}
        for k in range(self.n_samples):
            xi[k]={}
            for f in range(self.n_features):
                xi[k][f]=m.addVar(vtype=GRB.BINARY,name="abs[%d][%d]"%(k,f))

        feat ={}
        for k in range(self.n_samples):
            feat[k]={}
            for j in range(self.T_original):
                feat[k][j]=m.addVar(vtype=GRB.BINARY,name="feat[%d][%d]"%(k,j))

        for k in range(self.n_samples):
            for f in range(self.n_features):
                m.addConstr(-xi[k][f]<=self.X_.iloc[k,f]-x_l[k][f])
                m.addConstr(self.X_.iloc[k,f]-x_l[k][f]<=xi[k][f])

        
        #Add constraint to see if original feature is used
        for k in range(self.n_samples):
            for l in range(self.T_original): 
                m.addConstr(len(self.T_corr[l])*feat[k][l]>= gp.quicksum(xi[k][j] for j in self.T_corr[l]),name="SparseOriginalFeature")


        #Add max features constraint
        for k in range(self.n_samples):
            m.addConstr(gp.quicksum(feat[k][h] for h in range(self.T_original))<=self.Tmax_, name="MaxFeaturesCons")

        #Add constraint ensuring the one-hot encoding

        for k in range(self.n_samples):
            for l in self.T_c:
                m.addConstr(gp.quicksum(x_l[k][h] for h in self.T_corr[l])==1, name="Onehot")
     
       

        #ensure is a positive sample
        if clas_model[0]=='LR':
            w=clas_model[1].coef_
            b=clas_model[1].intercept_   
            for k in range(self.n_samples):
                m.addConstr(gp.quicksum(w[0][f]*x_l[k][f] for f in range(self.n_features))+b[0]>=0)

        elif clas_model[0]=='NN':
            nn_regression=clas_model[1]
            for k in range(self.n_samples):
                add_predictor_constr(m, nn_regression, x_l[k], y_clas[k])
                m.addConstr(y_clas[k][1]-y_clas[k][0]>=1e-3)

        m.optimize()

        counterf_init={}
        for k in range(self.n_samples):
            counterf_init[k]=[]
            for f in range(self.n_features):
                counterf_init[k].append(x_l[k][f].getAttr(GRB.Attr.X))

        list_counterf=list(counterf_init.values())
        
        unique_counterf = [element for index, element in enumerate(list_counterf) if element not in list_counterf[:index]]

        features_used_init={}
        for k in range(self.n_samples):
            features_used_init[k]=[]
            for f in range(self.T_original):
                if feat[k][f].getAttr(GRB.Attr.X)==1:
                    features_used_init[k].append(f)

        features_used_init_def={}
        for i, item in enumerate(list_counterf):
            for j,element in enumerate(unique_counterf):
                if item==element:
                    features_used_init_def[j]=features_used_init[i]

        A_init = np.zeros((len(list_counterf), len(unique_counterf)), dtype=int)

        for i, item in enumerate(list_counterf):
            for j, element in enumerate(unique_counterf):
                if item == element:
                    A_init[i, j] = 1

        sol_init=[]
        sol_init.append(unique_counterf)
        sol_init.append(A_init)
        sol_init.append(features_used_init_def)

        return sol_init

    def generate_mp(self, X, sol_init):
        """ Generates the master problem model (RMP) and initializes the primal
        and dual solutions.
        Parameters
        ----------
        X : ndarray, shape (self.n_samples, self.n_features)
            The input.
       
        """
        if self.generated_:
            return

        self.X_ = X

        self.counterf_init =  sol_init[0]
        self.A_= sol_init[1]
        self.features_init=sol_init[2]

        

       

        self.n_samples=self.X_.shape[0]

        # Initial solution
        self.n_counterf = len(self.counterf_init) 

        self.x = []
        for k in range(self.n_counterf):
            self.x.append(self.model_master.addVar(name="count[%d]"%k))

        self.counterf_vars_ = [None]*self.n_counterf
        for k in range(self.n_counterf):
            self.counterf_vars_[k] =self.x[k].index 
            self.counterf_dict_[self.counterf_vars_[k]]= self.counterf_init[k] 
            self.features_dict_[self.counterf_vars_[k]]=self.features_init[k]
        
        for col_index, col in enumerate(zip(*self.A_)):
            row_indices = [i for i, value in enumerate(col) if value == 1]
            if row_indices:
                self.assignment_dict_[col_index] = row_indices


            

        #objective

        self.model_master.setObjective(gp.quicksum(self.x[k] for k in range(self.n_counterf)), GRB.MINIMIZE)
        #self.model.update()
      
        
        #Add assignment constraint
        self.assignConst = []
        for i in range(self.n_samples):
            self.assignConst.append(self.model_master.addConstr(gp.quicksum(self.x[j] for j in range(self.n_counterf) if self.A_[i,j]==1)>= 1, name="AssignConst[%d]"%i))
       
        

        self.generated_ = True

        self.it_prueba=0

        

    def add_column(self, counterf, A_newcol,features_col):
        """ Adds the given column to the master problem model.
        Parameters
        ----------
        column : object,
            The column to be added to the RMP.
        """
        assert self.generated_ 

        if len(counterf) == 0:
            return False

        

        self.columns_added.append(A_newcol)
        
        #adding the column
        #print('Adding column')
        newCol = gp.Column()
        newCol.addTerms(A_newcol, self.assignConst)


        #adding the new variable
        self.n_counterf += 1
        id=self.n_counterf-1
        new_column_var = self.model_master.addVar(obj=1, name="count[%d]" % id,column=newCol)
        self.x.append(new_column_var)
        self.counterf_vars_.append(new_column_var.index)
        self.counterf_dict_[new_column_var.index] = counterf  
        self.features_dict_[new_column_var.index] = features_col
        
    

        #update the assignment dictionary
        self.assignment_dict_[new_column_var.index]=[index for index, element in enumerate(A_newcol) if element == 1]
      
       
       
        
        return True

    def solve_rmp(self, solver_params=''):
        """ Solves the RMP with given solver params.
        Returns the dual costs.
        Parameters
        ----------
        solver_params : string, default='',
            The solver parameters for solving the RMP.
        """
        assert self.generated_

        self.prev_rmp_objective=self.rmp_objective

        # solve
        if 'timelimit' in solver_params: 
            time_limit=solver_params['timelimit']
            self.model_master.setParam('TimeLimit', time_limit)
        self.model_master.optimize()

        self.rmp_objective=self.model_master.ObjVal

        #update the solutions dictionary
        for v in self.model_master.getVars():
            self.selected_dict_[v.index]=v.x

        
        
        # Dual costs
        y_duals=[]
        for c in self.model_master.getConstrs():
                if "AssignConst" in c.ConstrName:
                    y_duals.append(c.Pi)
               
        
        return y_duals

    def solve_ip(self, solver_params=''):
        """Solves the integer RMP with given solver params.
        Returns True if the explanation is generated.
        Parameters
        ----------
        solver_params : string, default='',
            The solver parameters for solving the integer RMP.
        """
        assert self.generated_

        #change variables to integer

        for k in range(self.n_counterf):
            self.x[k].vtype=GRB.BINARY

        #solve
        
        self.model_master.optimize()

        result_status = self.model_master.Status

        has_solution = (
            result_status == gp.GRB.OPTIMAL or
            result_status == gp.GRB.SUBOPTIMAL or
            result_status == gp.GRB.TIME_LIMIT)
        assert has_solution

        
        #update the solutions dictionary
        for v in self.model_master.getVars():
            self.selected_dict_[v.index]=v.x


        return True

    def rmp_objective_improved(self):

        if self.model_master.Status == gp.GRB.OPTIMAL:
            return self.prev_rmp_objective <self.rmp_objective
        else:
            False

   
class CFSubProblem():
    """The  subproblem for global counterfactuals.

    Parameters
    ----------
    Tmax : int,
        Maximum features used in the explanation

    Attributes
    ----------
    X_ : ndarray, shape (self.n_samples, self.n_features)
        The input passed during :meth:`generate_columns`. The inputs should
        only contain values in {0,1}.
    T_original : number of original features
    T_corr: dictionary Tcorr[l]: 
           set of indices of features in extended features (self.n_features) that 
           correspond to original feature l 
    T_c : list of indices of categorical or continuos features

    solver_ : Gurbipy
        The solver used for solving the subproblem.
    z_vars_ : list(int),
        Stores the indices of z variables.
    d_vars_ : list(int),
        Stores the indices of d variables.
    vgamma_: list(int),
        Stores the indices of v_h^gamma variables.
    generated_ : boolean,
        True when the master problem model has been generated. False otherwise.
    """

    def __init__(self, Tmax):

        self.Tmax_ = Tmax
        self.generated_ = False

        # Vars
        self.d_vars_ = None
        self.z_vars_ = None
        self.vgamma_= None

        self.model_pricing = gp.Model('PricingP')

    def create_submip(self, y_duals,clas_model):
        """Creates the model for the subproblem. This should be called only
        once for a given problem.
        Parameters
        ----------
      
        cs_duals : list(float),
            dual costs of covering of instances.
        """
        assert not self.generated_, "SP is already created."
        

        self.n_features = self.X_.shape[1]
        self.n_samples = self.X_.shape[0]

        #construct the decision variables

        #decision variable for feature used
        self.d = []
        for j in range(self.n_features):
            self.d.append(self.model_pricing.addVar(vtype=GRB.BINARY, name="d[%d]"%j))
        
        self.d_vars_ = [None]*self.n_features
        for j in range(self.n_features):
           self.d_vars_[j]=self.d[j].index

        #decision variable for original feature used
        self.feat =[]
        for j in range(self.T_original):
            self.feat.append(self.model_pricing.addVar(vtype=GRB.BINARY, name="f[%d]"%j))

        #construct decision variables for assigment
        self.z = []
        for i in range(self.n_samples):
            self.z.append(self.model_pricing.addVar(vtype=GRB.BINARY, name= "z[%d]"%i))

        self.z_vars_ = [None]*self.n_samples
        for i in range(self.n_samples):
           self.z_vars_[i]=self.z[i].index
  

        #construct the counterfactual

        self.v=self.model_pricing.addMVar(self.n_features, vtype=GRB.BINARY, name="v")

        #the class of the counterf
        self.y_clas=self.model_pricing.addMVar(2, lb=-gp.GRB.INFINITY, name="y_clas")

        #self.v =[]
        #for j in range(self.n_features): 
        #    self.v.append(self.model_pricing.addVar(vtype=GRB.BINARY,name="v[%d]"%j))

        #self.v_vars_ = [None]*self.n_features
        #for j in range(self.n_features):
        #   self.v_vars_[j]=j

        #decision variables for the absolute value
        self.change={}
        for i in range(self.n_samples):
            self.change[i]=[]
            for j in range(self.n_features):
                self.change[i].append(self.model_pricing.addVar(vtype=GRB.BINARY,name="change[%d]"%i))
    
        

        #Set objective
        

        
        self.model_pricing.setObjective(gp.quicksum(y_duals[i]*self.z[i] for i in range(self.n_samples)), GRB.MAXIMIZE)
        self.model_pricing.update()

        
        

        # Constraints

        for i in range(self.n_samples):
            for j in range(self.n_features):
                self.model_pricing.addConstr(-self.change[i][j]<=self.X_.iloc[i,j]-self.v[j])
                self.model_pricing.addConstr(self.X_.iloc[i,j]-self.v[j]<=self.change[i][j])

        #Add global sparsity constraint:
        for i in range(self.n_samples):
            for j in range(self.n_features):
                self.model_pricing.addConstr(self.d[j] >= self.change[i][j]*self.z[i], name="GlobalSparsityCons")

        #Add constraint to see if original feature is used
        for l in range(self.T_original): 
            self.model_pricing.addConstr(len(self.T_corr[l])*self.feat[l]>= gp.quicksum(self.d[j] for j in self.T_corr[l]),name="SparseOriginalFeature")


        #Add max features constraint
        self.model_pricing.addConstr(gp.quicksum(self.feat[h] for h in range(self.T_original))<=self.Tmax_, name="MaxFeaturesCons")

        #Add constraint ensuring the one-hot encoding

        for l in self.T_c:
            self.model_pricing.addConstr(gp.quicksum(self.v[h] for h in self.T_corr[l])==1, name="Onehot")


        
      

        #Add constr ensuring that v defines a positive sample
        #e.g. logistic regresion or nn

        if clas_model[0]=='LR':
            w=clas_model[1].coef_
            b=clas_model[1].intercept_   
            self.model_pricing.addConstr(gp.quicksum(w[0][f]*self.v[f] for f in range(self.n_features))+b[0]>=0, "score")

        elif clas_model[0]=='NN':
            nn_regression=clas_model[1]
            add_predictor_constr(self.model_pricing, nn_regression, self.v, self.y_clas)
            self.model_pricing.addConstr(self.y_clas[1]-self.y_clas[0]>=0)



        self.generated_=True

    def update_objective(self, y_duals):
        """Updates the objective of the generated subproblem. This can be
        called only after the create_submip method has been called.
        Parameters

        
        ----------
      
        cs_duals : list(float),
            dual costs of master problem.
        """
        assert self.generated_, "Subproblem not generated."
        
        
        self.model_pricing.setObjective(gp.quicksum(y_duals[i]*self.z[i] for i in range(self.n_samples)), GRB.MAXIMIZE)
        self.model_pricing.update()

    def additional_mip(self,counterf,A_newcol,params="",pool_milp=False):

        

        cfes_aux=[]
        assigns_aux=[]
        feats_aux=[]
        objs_aux=[]
        self.pool_milp=pool_milp

        #solve the pricing model with fixed explanation or assignment and get another column

        #fix the explanation

        model_pricing_aux1=self.model_pricing.copy()
        if self.pool_milp==False:
            model_pricing_aux1.setParam('PoolSolutions',1)


        explan_vars=[var for var in model_pricing_aux1.getVars() if var.VarName.startswith("v")]


        for var, value in zip(explan_vars, counterf):
            var.lb = value
            var.ub = value

        assign_vars=[var for var in model_pricing_aux1.getVars() if var.VarName.startswith("z[")]
        features_vars=[var for var in model_pricing_aux1.getVars() if var.VarName.startswith("f[")]

        model_pricing_aux1.addConstr(gp.quicksum(var + value - 2 * var * value for var, value in zip(assign_vars, A_newcol)) >= 1, "DifferentSolution")

        model_pricing_aux1.optimize()

        if model_pricing_aux1.status == GRB.OPTIMAL:
            Anewcol_aux1= model_pricing_aux1.getAttr(GRB.Attr.X, assign_vars)
            counterf_aux1=model_pricing_aux1.getAttr(GRB.Attr.X,explan_vars)
            features_used_aux1= [f for f in range(self.T_original) if model_pricing_aux1.getAttr(GRB.Attr.X,features_vars)[f]==1]
            cfes_aux.append(counterf_aux1)
            assigns_aux.append(Anewcol_aux1)
            feats_aux.append(features_used_aux1)
            objs_aux.append(model_pricing_aux1.ObjVal)


        #fix the assignment
        model_pricing_aux2=self.model_pricing.copy()
        if self.pool_milp==False:
            model_pricing_aux2.setParam('PoolSolutions', 1)

        assign_vars2=[var for var in model_pricing_aux2.getVars() if var.VarName.startswith("z[")]
        explan_vars2=[var for var in model_pricing_aux2.getVars() if var.VarName.startswith("v")]
        features_vars2=[var for var in model_pricing_aux2.getVars() if var.VarName.startswith("f[")]

        for var, value in zip(assign_vars2, A_newcol):
            var.lb = value
            var.ub = value


        model_pricing_aux2.addConstr(gp.quicksum(var + value - 2 * var * value for var, value in zip(explan_vars2, counterf)) >= 1, "DifferentSolution")

        model_pricing_aux2.optimize()

        if model_pricing_aux2.status == GRB.OPTIMAL:
            Anewcol_aux2= model_pricing_aux2.getAttr(GRB.Attr.X, assign_vars2)
            counterf_aux2=model_pricing_aux2.getAttr(GRB.Attr.X,explan_vars2)
            features_used_aux2= [f for f in range(self.T_original) if model_pricing_aux2.getAttr(GRB.Attr.X,features_vars2)[f]==1]
            cfes_aux.append(counterf_aux2)
            assigns_aux.append(Anewcol_aux2)
            feats_aux.append(features_used_aux2)
            objs_aux.append(model_pricing_aux2.ObjVal)

     

        return cfes_aux,assigns_aux,feats_aux,objs_aux

                    



    def generate_columns(self, X, sets_T, y_duals,clas_model, params="",pool=False,auxiliar_milp=False,pool_milp=False,pool_obj1=False):
        """Generates the new columns to be added to the RMP.
        In this case instead of directly generating the coefficients, this
        method returns the list of generated clauses. The Master problem can
        find the coefficients from it.

        Parameters
        ----------
        X : ndarray, shape (self.n_samples, self.n_features)
            The input. The inputs should only contain values in {0,1}.
       
        dual_costs : list(float)
            Dual costs of constraint
        params : string, default=""
            Solver parameters.
        """
        self.X_ = X
        y_dual =y_duals

        self.T_original=sets_T[0]
        self.T_corr=sets_T[1]
        self.T_c=sets_T[2]

        if 'threshold' in params:
            self.threshold=params['threshold']
        else:
            self.threshold=None

        self.pool=pool
        self.pool_milp=pool_milp
        self.pool_obj1=pool_obj1
        self.auxiliar_milp=auxiliar_milp


        # We assume the positive dual cost in this model.
        #for y in y_dual:
        #    assert y >= 0, "Negative y dual"

        for y in y_dual:
            if y<0:
                print('Negative dual')

        if self.generated_:
            print('Updating Objective')
            self.update_objective(y_duals)
        else:
            print('Creating pricing MIP')
            self.create_submip(y_duals,clas_model)

        # Solve sub problem
        if 'timelimit' in params: 
            time_limit=params['timelimit']
            self.model_pricing.setParam('TimeLimit', time_limit)
        

        def stop_subproblem(model,where):
            if where==gp.GRB.Callback.MIP:
                current_obj_value=model.cbGet(gp.GRB.Callback.MIP_OBJBST)
                if current_obj_value>=self.threshold:
                    print(f"Stopping optimization with objective value {current_obj_value} reached.")
                    model.terminate()



        ##parameter to choose if pool solutions have same objective function (gap 0)
        ## or it is allowed to be worse

        #keep more solutions
        #there are solutions with same objective but different explanation/assigment
        #there are feasible solutions with objVal>1 but worse obj function
        #keep even the ones with ObjVal<1 
        
        if self.pool==True: 
            # Limit how many solutions to collect
            self.model_pricing.setParam(GRB.Param.PoolSolutions, 5)
            # Limit the search space by setting a gap for the worst possible solution
            # that will be accepted
            self.model_pricing.setParam(GRB.Param.PoolGap, 0)
            # do a systematic search for the k-best solutions
            self.model_pricing.setParam(GRB.Param.PoolSearchMode, 2)

            




        if self.threshold==None:
            self.model_pricing.optimize()
        else:
            self.model_pricing.optimize(callback=stop_subproblem)
        #self.model_pricing.write('pricing_pr.lp')

        # Empty column is always feasible.
        #has_solution = (result_status == pywraplp.Solver.OPTIMAL or
        #                result_status == pywraplp.Solver.FEASIBLE)
        #assert has_solution

        features_used= [f for f in range(self.T_original) if self.feat[f].getAttr(GRB.Attr.X) == 1]
        A_newcol=[var.x for var in self.model_pricing.getVars() if var.VarName.startswith("z[") and var.VarName.endswith("]")]
        counterf=[var.x for var in self.model_pricing.getVars() if var.VarName.startswith("v")]

        
        if self.auxiliar_milp==True:
            cfes_aux,assigns_aux,feats_aux,objs_aux=self.additional_mip(counterf,A_newcol,pool_milp=self.pool_milp)
        else:
            cfes_aux=[]
            assigns_aux=[]
            feats_aux=[]
            objs_aux=[]
        
    
        if self.pool==True:
                print('Retrieving more solutions')
                nSolutions = self.model_pricing.SolCount
                #gaps=[]
                cfe=[]
                assignments=[]
                objs_val=[]
                feats=[]
                for i in range(nSolutions):
                    self.model_pricing.setParam(GRB.Param.SolutionNumber, i)
                    #solution_gap = self.model_pricing.PoolObjVal - self.model_pricing.ObjVal 
                    obj_function= self.model_pricing.PoolObjVal
                    #gaps.append(solution_gap) 
                    #TO DO: differenciate between columns with gap 0 and negative gaps 
                    A_newcol_new_aux=[var for var in self.model_pricing.getVars() if var.VarName.startswith("z[")]
                    A_newcol_new= self.model_pricing.getAttr(GRB.Attr.Xn, A_newcol_new_aux)
                    counterf_new_aux=[var for var in self.model_pricing.getVars() if var.VarName.startswith("v")]
                    counterf_new=self.model_pricing.getAttr(GRB.Attr.Xn,counterf_new_aux)
                    features_new_aux=[var for var in self.model_pricing.getVars() if var.VarName.startswith("f[")]
                    features_used_new= [f for f in range(self.T_original) if self.model_pricing.getAttr(GRB.Attr.Xn,features_new_aux)[f]==1]
                    
                    new_element=(A_newcol_new,counterf_new,features_used_new)

                    

                    if A_newcol_new==A_newcol and counterf_new == counterf and features_used_new==features_used:
                        #print('Nothing new')
                        pass

                    elif self.auxiliar_milp==True and new_element in zip(assigns_aux,cfes_aux,feats_aux):
                        #print('Repeated from MIP')
                        pass


                    elif new_element not in zip(assignments,cfe,feats):
                        print('new solution')
                        cfe.append(counterf_new)
                        assignments.append(A_newcol_new)
                        feats.append(features_used_new)
                        objs_val.append(obj_function)
                    
                    else:
                        #print('solution repeated')
                        pass
        
        #get a new column only if it has a negative reduced cost

        col_cfe=[]
        col_A=[]
        col_feat=[]

        #saving columns that have objVal<1
        pool_cfe=[]
        pool_A=[]
        pool_feat=[]
            

        if self.model_pricing.objVal > 1:   
            
            col_cfe.append(counterf)
            col_A.append(A_newcol)
            col_feat.append(features_used)

        

            if self.auxiliar_milp==True:
                col_cfe.extend([cf for cf, obj in zip(cfes_aux, objs_aux) if obj > 1])
                col_A.extend([assign for assign, obj in zip(assigns_aux, objs_aux) if obj > 1])
                col_feat.extend([f for f, obj in zip(feats_aux, objs_aux) if obj > 1])

                pool_cfe.extend([cf for cf, obj in zip(cfes_aux, objs_aux) if obj <= 1])
                pool_A.extend([assign for assign, obj in zip(assigns_aux, objs_aux) if obj <= 1])
                pool_feat.extend([f for f, obj in zip(feats_aux, objs_aux) if obj <= 1])

            if self.pool==True:
                #print('addind more columns obj val >1')

                col_cfe.extend([cf for cf, obj in zip(cfe, objs_val) if obj > 1])
                col_A.extend([assign for assign, obj in zip(assignments, objs_val) if obj > 1])
                col_feat.extend([f for f, obj in zip(feats, objs_val) if obj > 1])

                pool_cfe.extend([cf for cf, obj in zip(cfe, objs_val) if obj <= 1])
                pool_A.extend([assign for assign, obj in zip(assignments, objs_val) if obj <= 1])
                pool_feat.extend([f for f, obj in zip(feats, objs_val) if obj <= 1])

                 
            

            return [col_cfe,col_A,col_feat],[pool_cfe,pool_A,pool_feat]
            
        else:
            print('No columns generated.')
            return [col_cfe,col_A,col_feat],[pool_cfe,pool_A,pool_feat]
    




class GlobalCounterfColGen():

    """Binary classifier using boolean decision rule generation method.

    Parameters
    ----------
    max_iterations : int, default=-1
        Maximum column generation iterations. Negative values removes the
        iteration limit and the problem is solved till optimality.
    Tmax : int,default=3,
        A parameter used for controlling the overall sparsity of the explanation.
    rmp_solver_params: string, default = "",
        Solver parameters for solving restricted master problem (rmp).
    master_ip_solver_params: string, default = "",
        Solver parameters for solving the integer master problem.
    subproblem_params: list of strings, default = [""],
        Parameters for solving the subproblem.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
   
    """


    def __init__(self, max_iterations=-1,
                 time_limit=-1,
                 Tmax=3, 
                 counterfactual_instance=True,
                 pool=False,
                 pool_obj1=False,
                 auxiliar_milp=False,
                 pool_milp=False,
                 #master_problem=CFMasterProblem(Tmax=2),
                 #subproblems=[[CFSubProblem(Tmax=2)]],
                 rmp_is_ip=True,
                 rmp_solver_params="",
                 master_ip_solver_params="",
                 subproblem_params=[[""]]):
        self.Tmax_=Tmax
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.pool = pool
        self.auxiliar_milp=auxiliar_milp
        self.pool_milp=pool_milp
        self.pool_obj1 = pool_obj1
        self.master_problem = CFMasterProblem(Tmax)
        self.counterfactual_instance=counterfactual_instance
        if self.counterfactual_instance == True:
            self.subproblems = [[CFSubProblem(Tmax)]]
        else: 
            self.subproblems = [[CFSubProblem_2(Tmax)]] #add_onehot
        self.rmp_is_ip = rmp_is_ip
        self.rmp_solver_params = rmp_solver_params
        self.master_ip_solver_params = master_ip_solver_params
        self.subproblem_params = subproblem_params
       
        
        
    def fit(self,X,sets_T,clas_model):
        t_start = time()
        has_time_limit = self.time_limit > 0
    # Step 1: Generate initial solutions or data
        counterf_init = self.master_problem.init_sol(X, sets_T, clas_model)
        init_time=time()-t_start
        init_start=time()
        print('Number of initial counterfactuals: '+str(len(counterf_init[0])))
        # Step 2: Initiate the master and subproblem
        self.master_problem.generate_mp(X, counterf_init)

        self.performed_iter_=0
        self.num_improving_iter_=0 #add function that checks wheter the obj has improved
        self.mp_optimal_ = False
        self.time_limit_reached_ = False
        self.num_col_added_sp_ = []
        self.time_spent_sp_ = []
        self.time_add_col_ = 0.0
        self.time_spent_master_ = 0.0
        for level in range(len(self.subproblems)):
            self.num_col_added_sp_.append([0]*len(self.subproblems[level]))
            self.time_spent_sp_.append([0.0]*len(self.subproblems[level]))
        
        self.iter=0
        
        same_column=False
        same_matrix=False
        pool_columns_=[[],[],[]]
        while True:

            if self.max_iterations>0 and self.iter>self.max_iterations:
                break
    
            self.iter+=1
            print("Iteration number: ", self.iter)
            print("Time elapsed: ", time() - t_start)
            master_time_start = time()
            self.time_spent_master_ += time() - master_time_start
            if self.master_problem.rmp_objective_improved():
                self.num_improving_iter_ +=1

            # Step 3: Dual costs
            y_duals = self.master_problem.solve_rmp(solver_params=self.rmp_solver_params)
            
            #see before if columns in the pool now have obj >1 and add them directly
            if self.pool_obj1==True:
                if pool_columns_[0]!=[]:
                    cfe_pool=pool_columns_[0]
                    Anew_pool=pool_columns_[1]
                    features_pool=pool_columns_[2]
                    added_pool=False
                    for i in range(len(cfe_pool)):
                        new_ObjVal=sum(y_duals[i]*Anew_pool[i] for i in range(self.n_samples))
                        if new_ObjVal>1:
                            print('adding pool column')
                            added_pool=True
                            col_add_start = time()
                            col_added = self.master_problem.add_column(cfe_pool[i],Anew_pool[i],features_pool[i])
                            self.num_col_added_sp_[sp_level][sp_ind] += 1 if col_added else 0
                            self.time_add_col_ += time() - col_add_start

                    
                    if added_pool:
                        y_duals=self.master_problem.solve_rmp(solver_params=self.rmp_solver_params)


           
            #Step 4: Generate columns and add column
            rmp_updated=False
            for sp_level in range(len(self.subproblems)):  
                for sp_ind in range(len(self.subproblems[sp_level])):
                    self.time_elapsed_ = time() - t_start
                    if has_time_limit and self.time_elapsed_ > self.time_limit:
                        self.time_limit_reached_ = True
                        break
                    sp_time_start = time()
                    generated_columns_, pool_columns_ = self.subproblems[sp_level][sp_ind].generate_columns(
                        X, sets_T, y_duals,clas_model,params=self.subproblem_params,
                        pool=self.pool,auxiliar_milp=self.auxiliar_milp,pool_milp=self.pool_milp,pool_obj1=self.pool_obj1)
                    sp_time_end = time()
                    self.time_spent_sp_[sp_level][sp_ind] += sp_time_end - sp_time_start
                    cfe_column=generated_columns_[0]
                    Anewcol=generated_columns_[1]
                    features_col=generated_columns_[2]

                    if len(cfe_column)==1:##TO DO,: generalize this (maybe just see the first one)
                        if self.iter>2 and cfe_column in list(self.master_problem.counterf_dict_.values())[len(counterf_init[0]):] and Anewcol in self.master_problem.columns_added:
                            same_column=True
                        
                            break
                    col_add_start = time()
                    for i in range(len(cfe_column)):
                        col_added = self.master_problem.add_column(cfe_column[i],Anewcol[i],features_col[i])
                        self.num_col_added_sp_[sp_level][sp_ind] += 1 if col_added else 0
                        rmp_updated = rmp_updated or col_added
                    self.time_add_col_ += time() - col_add_start
                    if rmp_updated:
                        break
                self.time_elapsed_ = time() - t_start
                if self.time_limit_reached_:
                    print("Time limit reached!")
                    break
                if rmp_updated:
                    break
                if same_column:
                    print('same column added')
                    break

            if same_column:
                print('same column was generated')
                break
           

            if not rmp_updated:
                print("RMP not updated. exiting the loop.")
                if not self.time_limit_reached_:
                    self.mp_optimal_ = True
                break

        self.master_problem.solve_rmp()
        objective_relaxed=self.master_problem.model_master.ObjVal
        self.gap=100
        self.time_spent_master_ip_ = 0.0
        if self.rmp_is_ip:
            master_ip_start_time = time()
            solved = self.master_problem.solve_ip(self.master_ip_solver_params)
            self.time_spent_master_ip_ += time() - master_ip_start_time
            objective_integer=self.master_problem.model_master.ObjVal
            self.gap=(objective_integer-objective_relaxed)/objective_integer
            assert solved, "RMP integer program couldn't be solved."

        
        self.selected_sol=self.master_problem.selected_dict_

        self.counterf_sol = {key: self.master_problem.counterf_dict_[key] for key, value in self.selected_sol.items() if value == 1}
        self.assignment_sol={key: self.master_problem.assignment_dict_[key] for key, value in self.counterf_sol.items()}
        self.features_sol={key: self.master_problem.features_dict_[key] for key, value in self.selected_sol.items() if value ==1}

        self.time_elapsed_ = time() - t_start
        return self.counterf_sol, self.assignment_sol, self.features_sol


#if im interested in the explanation but not the instance
class CFSubProblem_2():
    """The  subproblem for global counterfactuals.

    Parameters
    ----------
    Tmax : int,
        Maximum features used in the explanation

    Attributes
    ----------
    X_ : ndarray, shape (self.n_samples, self.n_features)
        The input passed during :meth:`generate_columns`. The inputs should
        only contain values in {0,1}.
    solver_ : Gurbipy
        The solver used for solving the subproblem.
    z_vars_ : list(int),
        Stores the indices of z variables.
    d_vars_ : list(int),
        Stores the indices of d variables.
    vgamma_: list(int),
        Stores the indices of v_h^gamma variables.
    generated_ : boolean,
        True when the master problem model has been generated. False otherwise.
    """

    def __init__(self, Tmax):

        self.Tmax_ = Tmax
        self.generated_ = False

        # Vars
        self.d_vars_ = None
        self.z_vars_ = None
        self.vgamma_= None

        self.model = gp.Model('PricingP')

    def create_submip(self, y_duals,clas_model):
        """Creates the model for the subproblem. This should be called only
        once for a given problem.
        Parameters
        ----------
      
        cs_duals : list(float),
            dual costs of covering of instances.
        """
        assert not self.generated_, "SP is already created."
        

        self.n_features = self.X_.shape[1]
        self.n_samples = self.X_.shape[0]

        #construct the decision variables

        #decision variable for feature used
        self.d = []
        for j in range(self.n_features):
            self.d.append(self.model.addVar(vtype=GRB.BINARY, name="d[%d]"%j))
        
        self.d_vars_ = [None]*self.n_features
        for j in range(self.n_features):
           self.d_vars_[j]=self.d[j].index

        #decision variable for original feature used
        self.feat =[]
        for j in range(self.T_original):
            self.feat.append(self.model.addVar(vtype=GRB.BINARY, name="f[%d]"%j))

        #construct decision variables for assigment
        self.z = []
        for i in range(self.n_samples):
            self.z.append(self.model.addVar(vtype=GRB.BINARY, name= "z[%d]"%i))

        self.z_vars_ = [None]*self.n_samples
        for i in range(self.n_samples):
           self.z_vars_[i]=self.z[i].index
  

        #construct a counterfactual for each negative sample
        
        self.v ={}
        for i in range(self.n_samples):
            self.v[i]=[]
            for j in range(self.n_features): 
                self.v[i].append(self.model.addVar(vtype=GRB.BINARY,name="v[%d][%d]" % (i, j)))

        self.delta={}
        for h in range(self.n_features):
            self.delta[h]={}
            for i in range(self.n_samples):
                self.delta[h][i]=[]
                for j in range(self.n_samples):
                    self.delta[h][i].append(self.model.addVar(vtype=GRB.BINARY,name ="delta[%d][%d][%d] % (h,i,j)"))
                    

        #decision variables for the absolute value
        self.change={}
        for i in range(self.n_samples):
            self.change[i]=[]
            for j in range(self.n_features):
                self.change[i].append(self.model.addVar(vtype=GRB.BINARY,name="change[%d][%d]"%(i,j)))
    
            
        #Set objective
        

        self.model.setObjective(1-gp.quicksum(y_duals[i]*self.z[i] for i in range(self.n_samples)), GRB.MINIMIZE)
        self.model.update()

        
        

        # Constraints

        for i in range(self.n_samples):
            for j in range(self.n_features):
                self.model.addConstr(-self.change[i][j]<=self.X_.iloc[i,j]-self.v[i][j])
                self.model.addConstr(self.X_.iloc[i,j]-self.v[i][j]<=self.change[i][j])


        #Add global sparsity constraint:
        for i in range(self.n_samples):
            for j in range(self.n_features):
                self.model.addConstr(self.d[j] >= self.change[i][j]*self.z[i], name="GlobalSparsityCons")

        #Add constraint to see if original feature is used
        for l in range(self.T_original): 
            self.model.addConstr(len(self.T_corr[l])*self.feat[l]>= gp.quicksum(self.d[j] for j in self.T_corr[l]),name="SparseOriginalFeature")


        #Add max features constraint
        self.model.addConstr(gp.quicksum(self.feat[h] for h in range(self.T_original))<=self.Tmax_, name="MaxFeaturesCons")

        #Add constraint ensuring the one-hot encoding

        for i in range(self.n_samples):
            for l in self.T_c:
                self.model.addConstr(gp.quicksum(self.v[i][h] for h in self.T_corr[l])==1, name="Onehot")



        #add constraint dh=1 and zi=zj=1, then vih=vjh
        for i in range(self.n_samples):
            for j in range(self.n_samples):
                for h in range(self.n_features):
                    self.model.addConstr(self.delta[h][i][j]<=self.z[i])
                    self.model.addConstr(self.delta[h][i][j]<=self.z[j])
                    self.model.addConstr(self.delta[h][i][j]<=self.d[h])
                    self.model.addConstr(self.delta[h][i][j]>=self.z[i]+self.z[j]+self.d[h]-2)
                    self.model.addConstr((self.v[i][h]-self.v[j][h])*self.delta[h][i][j]==0)

      

        #Add constr ensuring that v defines a positive sample
        #e.g. logistic regresion
        w=clas_model.coef_
        b=clas_model.intercept_   
        for i in range(self.n_samples):
            self.model.addConstr(gp.quicksum(w[0][f]*self.v[i][f] for f in range(self.n_features))+b[0]>=0, "score")


        self.generated=True

    def update_objective(self, X, y_duals):
        """Updates the objective of the generated subproblem. This can be
        called only after the create_submip method has been called.
        Parameters

        
        ----------
      
        cs_duals : list(float),
            dual costs of master problem.
        """
        assert self.generated_, "Subproblem not generated."
        self.n_features = self.X_.shape[1]
        self.n_samples = self.X_.shape[0]

        #Update objective
        self.model.setObjective(1-gp.quicksum(y_duals[i]*self.z[i] for i in range(self.n_samples)), GRB.MINIMIZE)
        self.model.update()

    def generate_columns(self, X, sets_T, y_duals,clas_model, params=""):
        """Generates the new columns to be added to the RMP.
        In this case instead of directly generating the coefficients, this
        method returns the list of generated clauses. The Master problem can
        find the coefficients from it.

        Parameters
        ----------
        X : ndarray, shape (self.n_samples, self.n_features)
            The input. The inputs should only contain values in {0,1}.
       
        dual_costs : list(float)
            Dual costs of constraint
        params : string, default=""
            Solver parameters.
        """
        self.X_ = X
        y_dual =y_duals
        self.T_original=sets_T[0]
        self.T_corr=sets_T[1]
        self.T_c=sets_T[2]

        # We assume the positive dual cost in this model.
        for y in y_dual:
            assert y >= 0, "Negative y dual"

        if self.generated_:
            self.update_objective(y_duals)
        else:
            self.create_submip(y_duals,clas_model)

        # Solve sub problem
        self.model.optimize()

        # Empty column is always feasible.
        #has_solution = (result_status == pywraplp.Solver.OPTIMAL or
        #                result_status == pywraplp.Solver.FEASIBLE)
        #assert has_solution

        #get a new column only if it has a negative reduced cost

        if self.model.objVal < -1e-6:
            #for v in self.model.getVars():
            #    print('%s %g' % (v.varName, v.x))
                    
            features_used=[v.x for v in self.model.getVars()[0:len(self.d)]]

            A_newcol=[v.x for v in self.model.getVars()[len(self.d):len(self.d)+len(self.z)]]

            counterf=[]
            for j in range(self.n_features):
                if features_used[j]==0:
                    counterf.append('NU')
                else:
                    index = next((i for i, value in enumerate(A_newcol) if value == 1), None)
                    counterf.append([v.x for v in self.model.getVars()[len(self.d)+len(self.z):len(self.z)+len(self.d)+len(self.v)] if '['+str(index)+']['+str(j)+']' in v.varName][0])

            
            
           
           
            
            return counterf, A_newcol
        else:
            print('No columns with reduced costs generated.')
            return [], []

