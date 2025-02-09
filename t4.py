from pulp import LpMinimize, LpProblem, LpVariable, lpSum
import pandas as pd

# Load dataset (Assuming cleaned_people_data.csv exists)
df = pd.read_csv("cleaned_people_data.csv")

# Sample job demand & salary data
job_demand = {"Engineer": 3, "Manager": 2, "Technician": 4}
job_salaries = {"Engineer": 60000, "Manager": 80000, "Technician": 50000}
budget = 400000  # Total budget limit

# Define LP Problem
model = LpProblem("Employee_Hiring_Optimization", LpMinimize)

# Define decision variables (number of employees to hire per job)
employees = {role: LpVariable(f"hire_{role}", lowBound=0, cat="Integer") for role in job_demand}

# Objective Function: Minimize total hiring cost
model += lpSum(employees[job] * job_salaries[job] for job in job_demand)

# Constraints: Meet job demand
for job in job_demand:
    model += employees[job] >= job_demand[job]

# Budget Constraint
model += lpSum(employees[job] * job_salaries[job] for job in job_demand) <= budget

# Solve the problem
model.solve()

# Print results
print("✅ Optimal Hiring Plan:")
for job in job_demand:
    print(f"Hire {int(employees[job].varValue)} {job}(s)")

print(f"Total Cost: ${sum(int(employees[job].varValue) * job_salaries[job] for job in job_demand)}")
