import plotly.express as px
import pandas as pd

df = pd.read_csv("hours_worked.csv")
print(df)

fig = px.scatter(df,
  x="Average hours worked per person employed",
  y="GDP per capita, PPP, constant 2011 USD",
  color="Continent",
  hover_name="Country Name",
  size="Population",
  size_max=80,
  log_x=True,
)

fig.show()