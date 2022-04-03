import time
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#import seaborn as sns
import pulp as p
#import random
#from itertools import combinations
#import winsound
#import os
#from openpyxl import Workbook

M=99999 # An Arbitrary Big Value

# Getting the Input
FE=100 # Fixed Emission
E=5 # Variable Emission
Q=500
#Q=float(input("Enter the Vehicle Capacity to be used for all the transportation links: "))
Customers=pd.read_excel("Locations and Distances with Supply, Demands & Factors.xlsx","Customers",index_col=0)
Branches=pd.read_excel("Locations and Distances with Supply, Demands & Factors.xlsx","Branches",index_col=0)
Distribution_Centres=pd.read_excel("Locations and Distances with Supply, Demands & Factors.xlsx","Distribution Centres",index_col=0)
Suppliers=pd.read_excel("Locations and Distances with Supply, Demands & Factors.xlsx","Suppliers",index_col=0)

# Creating the Sets for Image
Customers_Location={}
Branches_Location={}
Distribution_Centres_Location={}
Suppliers_Location={}

# Creating the Sets
Customer_Set=set()
Branch_Set=set()
DC_Set=set()
Supplier_Set=set()
Product_Set={1,2}

# Inputs Required
Demand={}
Supply={}
Branch_Fixed_Emission={}
DC_Fixed_Emission={}
Branch_Variable_Emission={}
DC_Variable_Emission={}
Branch_Capacity={}
DC_Capacity={}

for i, row in Customers.iterrows():
    value=(row["Latitude"],row["Longitude"])
    Customers_Location[i]=value
    for k in Product_Set:
        key=(i,k)
        superstring="Demand of Product "+str(k)
        value=row[superstring]
        Demand[key]=value
    Customer_Set.add(i)

for i, row in Branches.iterrows():
    value=(row["Latitude"],row["Longitude"])
    Branches_Location[i]=value
    value=row["Fixed Emissions per year due to Facility Maintenance"]
    Branch_Fixed_Emission[i]=value
    value=row["Variable Emissions per year due to Daily Operations"]
    Branch_Variable_Emission[i]=value
    for k in Product_Set:
        key=(i,k)
        superstring="Inventory Holding and Handling Capacity for Product "+str(k)+" for the specific period"
        value=row[superstring]
        Branch_Capacity[key]=value
    Branch_Set.add(i)

for i, row in Distribution_Centres.iterrows():
    value=(row["Latitude"],row["Longitude"])
    Distribution_Centres_Location[i]=value
    value=row["Fixed Emissions per year due to Facility Maintenance"]
    DC_Fixed_Emission[i]=value
    value=row["Variable Emissions per year due to Daily Operations"]
    DC_Variable_Emission[i]=value
    for k in Product_Set:
        key=(i,k)
        superstring="Inventory Holding and Handling Capacity for Product "+str(k)+" for the specific period"
        value=row[superstring]
        DC_Capacity[key]=value
    DC_Set.add(i)

for i, row in Suppliers.iterrows():
    value=(row["Latitude"],row["Longitude"])
    Suppliers_Location[i]=value
    for k in Product_Set:
        key=(i,k)
        superstring="Supply of Product "+str(k)
        value=row[superstring]
        Supply[key]=value
    Supplier_Set.add(i)

# Creating the Distance Matrices

Distance_S_DC={}
Distance_DC_B={}
Distance_B_C={}
Distance_S_B={}
Distance_DC_C={}

Data=pd.read_excel("Distances.xlsx","Suppliers to DCs",index_col=0)
for i in Supplier_Set:
    for j in DC_Set:
        Distance_S_DC[i,j]=Data.loc[i,j]

Data=pd.read_excel("Distances.xlsx","DCs to Branches",index_col=0)
for i in DC_Set:
    for j in Branch_Set:
        Distance_DC_B[i,j]=Data.loc[i,j]

Data=pd.read_excel("Distances.xlsx","Branches to Customers",index_col=0)
for i in Branch_Set:
    for j in Customer_Set:
        Distance_B_C[i,j]=Data.loc[i,j]

Data=pd.read_excel("Distances.xlsx","Suppliers to Branches",index_col=0)
for i in Supplier_Set:
    for j in Branch_Set:
        Distance_S_B[i,j]=Data.loc[i,j]

Data=pd.read_excel("Distances.xlsx","DCs to Customers",index_col=0)
for i in DC_Set:
    for j in Customer_Set:
        Distance_DC_C[i,j]=Data.loc[i,j]

# Formulating the Problem
prob=p.LpProblem("Facility Location developing Sustainable Practices and ensuring development of Green Supply Chain",p.LpMinimize)

x_S_DC=p.LpVariable.dicts('Supplier to DC Path',((i,j) for i in Supplier_Set for j in DC_Set),cat='Binary')
x_DC_B=p.LpVariable.dicts('DC to Branch Path',((i,j) for i in DC_Set for j in Branch_Set),cat='Binary')
x_B_C=p.LpVariable.dicts('Branch to Customer Path',((i,j) for i in Branch_Set for j in Customer_Set),cat='Binary')
x_S_B=p.LpVariable.dicts('Supplier to Branch Path',((i,j) for i in Supplier_Set for j in Branch_Set),cat='Binary')
x_DC_C=p.LpVariable.dicts('DC to Customer Path',((i,j) for i in DC_Set for j in Customer_Set),cat='Binary')

y_S_DC=p.LpVariable.dicts('Supplier to DC Product Flow',((i,j,k) for i in Supplier_Set for j in DC_Set for k in Product_Set),lowBound=0)
y_DC_B=p.LpVariable.dicts('DC to Branch Product Flow',((i,j,k) for i in DC_Set for j in Branch_Set for k in Product_Set),lowBound=0)
y_B_C=p.LpVariable.dicts('Branch to Customer Product Flow',((i,j,k) for i in Branch_Set for j in Customer_Set for k in Product_Set),lowBound=0)
y_S_B=p.LpVariable.dicts('Supplier to Branch Product Flow',((i,j,k) for i in Supplier_Set for j in Branch_Set for k in Product_Set),lowBound=0)
y_DC_C=p.LpVariable.dicts('DC to Customer Product Flow',((i,j,k) for i in DC_Set for j in Customer_Set for k in Product_Set),lowBound=0)

z_DC=p.LpVariable.dicts('Existance of DCs',(i for i in DC_Set),cat='Binary')
z_B=p.LpVariable.dicts('Existance of Branches',(i for i in Branch_Set),cat='Binary')


# Emission Minimization Objective Function
#prob+=p.lpSum(x[i,j,k]*C[i,j,k]*VS[k] for k in Vehicle_Types for i in Depot_and_Relief_Centres for j in Depot_and_Relief_Centres)+p.lpSum(x[0,j,k]*VC[k] for k in Vehicle_Types for j in Relief_Centres)

prob+=p.lpSum((x_S_DC[i,j]*FE*2+p.lpSum(E*y_S_DC[i,j,k] for k in Product_Set))*Distance_S_DC[(i,j)] for i in Supplier_Set for j in DC_Set) + p.lpSum((x_S_B[i,j]*FE*2+p.lpSum(E*y_S_B[i,j,k] for k in Product_Set))*Distance_S_B[(i,j)] for i in Supplier_Set for j in Branch_Set) + p.lpSum((x_DC_B[i,j]*FE*2+p.lpSum(E*y_DC_B[i,j,k] for k in Product_Set))*Distance_DC_B[(i,j)] for i in DC_Set for j in Branch_Set) + p.lpSum((x_DC_C[i,j]*FE*2+p.lpSum(E*y_DC_C[i,j,k] for k in Product_Set))*Distance_DC_C[(i,j)] for i in DC_Set for j in Customer_Set) + p.lpSum((x_B_C[i,j]*FE*2+p.lpSum(E*y_B_C[i,j,k] for k in Product_Set))*Distance_B_C[(i,j)] for i in Branch_Set for j in Customer_Set) + p.lpSum(z_B[i]*Branch_Fixed_Emission[i] for i in Branch_Set) + p.lpSum((p.lpSum(y_S_B[j,i,k] for j in Supplier_Set)+p.lpSum(y_DC_B[j,i,k] for j in DC_Set))*Branch_Variable_Emission[i] for i in Branch_Set for k in Product_Set) + p.lpSum(z_DC[i]*DC_Fixed_Emission[i] for i in DC_Set) + p.lpSum((p.lpSum(y_S_DC[j,i,k] for j in Supplier_Set for k in Product_Set))*DC_Variable_Emission[i] for i in DC_Set)

# Constraint 1
for k in Product_Set:
    for i in Supplier_Set:
        prob+=p.lpSum(y_S_DC[i,j,k] for j in DC_Set) + p.lpSum(y_S_B[i,j,k] for j in Branch_Set) <= Supply[(i,k)]

# Constraint 2
for k in Product_Set:
    for i in DC_Set:
        prob+=p.lpSum(y_DC_B[i,j,k] for j in Branch_Set) + p.lpSum(y_DC_C[i,j,k] for j in Customer_Set) <= p.lpSum(y_S_DC[j,i,k] for j in Supplier_Set)

# Constraint 3
for k in Product_Set:
    for i in Branch_Set:
        prob+=p.lpSum(y_B_C[i,j,k] for j in Customer_Set) <= p.lpSum(y_S_B[j,i,k] for j in Supplier_Set) + p.lpSum(y_DC_B[j,i,k] for j in DC_Set)

# Constraint 4
for k in Product_Set:
    for j in DC_Set:
        prob+=p.lpSum(y_S_DC[i,j,k] for i in Supplier_Set) <= DC_Capacity[(j,k)]

# Constraint 5
for k in Product_Set:
    for j in Branch_Set:
        prob+=p.lpSum(y_DC_B[i,j,k] for i in DC_Set) + p.lpSum(y_S_B[i,j,k] for i in Supplier_Set) <= Branch_Capacity[(j,k)]

# Constraint 6
for k in Product_Set:
    for j in Customer_Set:
        prob+=p.lpSum(y_B_C[i,j,k] for i in Branch_Set) + p.lpSum(y_DC_C[i,j,k] for i in DC_Set) == Demand[(j,k)]

# Constraint 7

for i in Supplier_Set:
    for j in DC_Set:
        prob+=p.lpSum(y_S_DC[i,j,k] for k in Product_Set) <= x_S_DC[i,j]*Q

for i in Supplier_Set:
    for j in Branch_Set:
        prob+=p.lpSum(y_S_B[i,j,k] for k in Product_Set) <= x_S_B[i,j]*Q

for i in DC_Set:
    for j in Branch_Set:
        prob+=p.lpSum(y_DC_B[i,j,k] for k in Product_Set) <= x_DC_B[i,j]*Q

for i in DC_Set:
    for j in Customer_Set:
        prob+=p.lpSum(y_DC_C[i,j,k] for k in Product_Set) <= x_DC_C[i,j]*Q

for i in Branch_Set:
    for j in Customer_Set:
        prob+=p.lpSum(y_B_C[i,j,k] for k in Product_Set) <= x_B_C[i,j]*Q

# Constraint 8
for i in Branch_Set:
    prob+=p.lpSum(x_S_B[j,i] for j in Supplier_Set) + p.lpSum(x_DC_B[j,i] for j in DC_Set) + p.lpSum(x_B_C[i,j] for j in Customer_Set) <= z_B[i]*M

# Constraint 9
for i in DC_Set:
    prob+=p.lpSum(x_S_DC[j,i] for j in Supplier_Set) + p.lpSum(x_DC_B[i,j] for j in Branch_Set) + p.lpSum(x_DC_C[i,j] for j in Customer_Set) <= z_DC[i]*M


# Solve the Problem using default CBC
start_time=time.time()
#status=prob.solve(p.PULP_CBC_CMD(maxSeconds=300, msg=1, gapRel=0))
status=prob.solve(p.PULP_CBC_CMD(timeLimit=99))
#status=prob.solve()
end_time=time.time()

#Plotting the Figure
plt.figure(figsize=(11,11))
for i in Suppliers_Location:
    plt.scatter(Suppliers_Location[i][1],Suppliers_Location[i][0],c='r',marker='s')
    plt.text(Suppliers_Location[i][1]+0.33,Suppliers_Location[i][0]+0.33,str(i))
for i in Distribution_Centres_Location:
    plt.scatter(Distribution_Centres_Location[i][1],Distribution_Centres_Location[i][0],c='m',marker='h')
    plt.text(Distribution_Centres_Location[i][1]+0.33,Distribution_Centres_Location[i][0]+0.33,str(i))
for i in Branches_Location:
    plt.scatter(Branches_Location[i][1],Branches_Location[i][0],c='c',marker='*')
    plt.text(Branches_Location[i][1]+0.33,Branches_Location[i][0]+0.33,str(i))
for i in Customers_Location:
    plt.scatter(Customers_Location[i][1],Customers_Location[i][0],c='g',marker="P")
    plt.text(Customers_Location[i][1]+0.33,Customers_Location[i][0]+0.33,str(i))
plt.title("Suppliers, Available Facilities and Customers")
plt.ylabel("Latitude")
plt.xlabel("Longitude")
plt.savefig("{}".format("Suppliers, Available Facilities and Customers"))

for k in Product_Set:
    plt.figure(figsize=(13,13))
    for i in Suppliers_Location:
        plt.scatter(Suppliers_Location[i][1],Suppliers_Location[i][0],c='r',marker='s')
        plt.text(Suppliers_Location[i][1]+0.33,Suppliers_Location[i][0]+0.33,str(i))
    for i in Distribution_Centres_Location:
        plt.scatter(Distribution_Centres_Location[i][1],Distribution_Centres_Location[i][0],c='m',marker='h')
        plt.text(Distribution_Centres_Location[i][1]+0.33,Distribution_Centres_Location[i][0]+0.33,str(i))
    for i in Branches_Location:
        plt.scatter(Branches_Location[i][1],Branches_Location[i][0],c='c',marker='*')
        plt.text(Branches_Location[i][1]+0.33,Branches_Location[i][0]+0.33,str(i))
    for i in Customers_Location:
        plt.scatter(Customers_Location[i][1],Customers_Location[i][0],c='g',marker="P")
        plt.text(Customers_Location[i][1]+0.33,Customers_Location[i][0]+0.33,str(i))

    arrow_S_DC = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='tab:purple')
    arrow_S_B = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='tab:brown')
    arrow_DC_B = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='tab:orange')
    arrow_DC_C = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='tab:green')
    arrow_B_C = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='tab:olive')

    for i in Supplier_Set:
        for j in DC_Set:
            if p.value(x_S_DC[i,j])==1:
                plt.annotate('', xytext=[Suppliers_Location[i][1], Suppliers_Location[i][0]], xy=[Distribution_Centres_Location[j][1], Distribution_Centres_Location[j][0]], arrowprops=arrow_S_DC)
                plt.text((Suppliers_Location[i][1]+Distribution_Centres_Location[j][1])/2, (Suppliers_Location[i][0]+Distribution_Centres_Location[j][0])/2, f'{p.value(y_S_DC[i,j,k])}',fontweight="bold")
        for j in Branch_Set:
            if p.value(x_S_B[i,j])==1:
                plt.annotate('', xytext=[Suppliers_Location[i][1], Suppliers_Location[i][0]], xy=[Branches_Location[j][1], Branches_Location[j][0]], arrowprops=arrow_S_B)
                plt.text((Suppliers_Location[i][1]+Branches_Location[j][1])/2, (Suppliers_Location[i][0]+Branches_Location[j][0])/2, f'{p.value(y_S_B[i,j,k])}',fontweight="bold")
    for i in DC_Set:
        for j in Branch_Set:
            if p.value(x_DC_B[i,j])==1:
                plt.annotate('', xytext=[Distribution_Centres_Location[i][1], Distribution_Centres_Location[i][0]], xy=[Branches_Location[j][1], Branches_Location[j][0]], arrowprops=arrow_DC_B)
                plt.text((Distribution_Centres_Location[i][1]+Branches_Location[j][1])/2, (Distribution_Centres_Location[i][0]+Branches_Location[j][0])/2, f'{p.value(y_DC_B[i,j,k])}',fontweight="bold")
        for j in Customer_Set:
            if p.value(x_DC_C[i,j])==1:
                plt.annotate('', xytext=[Distribution_Centres_Location[i][1], Distribution_Centres_Location[i][0]], xy=[Customers_Location[j][1], Customers_Location[j][0]], arrowprops=arrow_DC_C)
                plt.text((Distribution_Centres_Location[i][1]+Customers_Location[j][1])/2, (Distribution_Centres_Location[i][0]+Customers_Location[j][0])/2, f'{p.value(y_DC_C[i,j,k])}',fontweight="bold")
    for i in Branch_Set:
        for j in Customer_Set:
            if p.value(x_B_C[i,j])==1:
                plt.annotate('', xytext=[Branches_Location[i][1], Branches_Location[i][0]], xy=[Customers_Location[j][1], Customers_Location[j][0]], arrowprops=arrow_B_C)
                plt.text((Customers_Location[j][1]+Branches_Location[i][1])/2, (Customers_Location[j][0]+Branches_Location[i][0])/2, f'{p.value(y_B_C[i,j,k])}',fontweight="bold")
    
    plt.title("Flow Layer for Product Type "+str(k)+" on the corresponding layer "+str(k)+". The Total Emission as per the minimised objective is: "+str(p.value(prob.objective)))
    plt.ylabel("Latitude")
    plt.xlabel("Longitude")
    plt.savefig("{}".format("Flow Layer for Product Type "+str(k)+".png"))