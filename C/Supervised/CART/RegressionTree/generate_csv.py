import csv
import random
import math

# Set seed for reproducibility
random.seed(42)

# Generate CSV with categorical and numerical features
def generate_data_with_categorical():
    data = []
    header = ['num_feature', 'cat_feature', 'label']

    for i in range(100):
        # Numerical feature: random float between 0 and 100
        num_feature = round(random.uniform(0, 100), 2)

        # Categorical feature: random integer 0, 1, or 2
        cat_feature = random.randint(0, 2)

        # Label: based on some relationship with features
        # Add some noise to make it interesting
        label = int(num_feature * 0.5 + cat_feature * 10 + random.uniform(-5, 5))

        data.append([num_feature, cat_feature, label])

    return header, data

# Generate CSV with only numerical feature
def generate_data_numerical_only():
    data = []
    header = ['num_feature', 'label']

    for i in range(100):
        # Numerical feature: random float between 0 and 100
        num_feature = round(random.uniform(0, 100), 2)

        # Label: based on some relationship with feature
        # Using a non-linear relationship
        label = int(2 * num_feature + 0.1 * num_feature**2 + random.uniform(-10, 10))

        data.append([num_feature, label])

    return header, data

# Write categorical dataset
header1, data1 = generate_data_with_categorical()
with open('C:/Users/pstay/CLionProjects/MLfromScratch/C/Supervised/CART/RegressionTree/data_categorical.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header1)
    writer.writerows(data1)

print("Created data_categorical.csv with 100 rows")

# Write numerical only dataset
header2, data2 = generate_data_numerical_only()
with open('C:/Users/pstay/CLionProjects/MLfromScratch/C/Supervised/CART/RegressionTree/data_numerical.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header2)
    writer.writerows(data2)

print("Created data_numerical.csv with 100 rows")

