import pickle

# Classification Model
with open('data/processed/classification_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)

print("Classification Model:")
print(clf_model)
print(type(clf_model))
print(len(clf_model.estimators_))

print("\n----------------------\n")

# Salary Model
with open('data/processed/salary_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)

print("Salary Prediction Model:")
print(reg_model)
print(type(reg_model))