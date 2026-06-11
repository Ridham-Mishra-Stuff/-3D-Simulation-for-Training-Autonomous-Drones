import matplotlib.pyplot as plt

sessions = [1,2,3,4,5]

navigation = [60,70,80,89,96]

collision = [15,11,8,4,2]

mapping = [68,74,81,88,92]

plt.figure(figsize=(8,5))

plt.plot(
    sessions,
    navigation,
    marker='o'
)

plt.xlabel(
    "Training Session"
)

plt.ylabel(
    "Navigation Success (%)"
)

plt.title(
    "Navigation Performance Improvement"
)

plt.grid(True)

plt.savefig(
    "../images/navigation_performance.png",
    dpi=300
)

plt.close()
