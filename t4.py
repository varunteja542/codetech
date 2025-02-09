from pulp import LpMinimize, LpProblem, LpVariable, lpSum
import pandas as pd

df = pd.read_csv("cleaned_people_data.csv")

job_demand = {"Engineer": 3, "Manager": 2, "Technician": 4}
job_salaries = {"Engineer": 60000, "Manager": 80000, "Technician": 50000}
budget = 400000

model = LpProblem("Employee_Hiring_Optimization", LpMinimize)

employees = {role: LpVariable(f"hire_{role}", lowBound=0, cat="Integer") for role in job_demand}

model += lpSum(employees[job] * job_salaries[job] for job in job_demand)

for job in job_demand:
    model += employees[job] >= job_demand[job]

model += lpSum(employees[job] * job_salaries[job] for job in job_demand) <= budget

model.solve()

print("✅ Optimal Hiring Plan:")
for job in job_demand:
    print(f"Hire {int(employees[job].varValue)} {job}(s)")

print(f"Total Cost: ${sum(int(employees[job].varValue) * job_salaries[job] for job in job_demand)}")
