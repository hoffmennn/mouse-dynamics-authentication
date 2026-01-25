import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# načítanie dát
df = pd.read_csv("tr5.csv", skipinitialspace=True)
df.columns = df.columns.str.strip()

# základný štýl
sns.set(style="whitegrid")

plt.figure(figsize=(8, 6))

# 1. pohyb (Move) – tenká sivá čiara
move = df[df["state"] == "Move"]
plt.plot(move["x"], move["y"], color="gray", alpha=0.5, label="Move")



# 2. drag – hrubšia farebná čiara
drag = df[df["state"] == "Drag"]
plt.plot(drag["x"], drag["y"], color="tab:blue", linewidth=2, label="Drag")


plt.scatter(
    move["x"],
    move["y"],
    color="gray",
    s=12,
    alpha=0.7,
    label="Move points"
)

# 3. kliky (Pressed / Released)
clicks = df[df["state"].isin(["Pressed", "Released"])]
sns.scatterplot(
    data=clicks,
    x="x",
    y="y",
    hue="state",
    style="state",
    s=100,
    palette={"Pressed": "red", "Released": "green"},
    legend=True
)

# inverzia osi Y (súradnice obrazovky)
plt.gca().invert_yaxis()
plt.legend().remove()
plt.title("")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
