import pandas as pd
import numpy as np

# Load results
results_df = pd.read_csv("results_val.csv")

print("=" * 80)
print("VALIDATION RESULTS ANALYSIS")
print("=" * 80)

# Overall metrics
accuracy = (results_df["prediction"] == results_df["true_label"]).mean()
print(f"\n📊 Overall Accuracy: {accuracy:.2%}")

# Confusion matrix
tp = ((results_df["prediction"] == 1) & (results_df["true_label"] == 1)).sum()
tn = ((results_df["prediction"] == 0) & (results_df["true_label"] == 0)).sum()
fp = ((results_df["prediction"] == 1) & (results_df["true_label"] == 0)).sum()
fn = ((results_df["prediction"] == 0) & (results_df["true_label"] == 1)).sum()

print(f"\n📈 Confusion Matrix:")
print(f"   True Positives  (Correct CONSISTENT):     {tp}")
print(f"   True Negatives  (Correct CONTRADICTORY):  {tn}")
print(f"   False Positives (Wrong CONSISTENT):       {fp}")
print(f"   False Negatives (Wrong CONTRADICTORY):    {fn}")

# Precision, Recall, F1
if tp + fp > 0:
    precision = tp / (tp + fp)
else:
    precision = 0.0

if tp + fn > 0:
    recall = tp / (tp + fn)
else:
    recall = 0.0

if precision + recall > 0:
    f1 = 2 * (precision * recall) / (precision + recall)
else:
    f1 = 0.0

print(f"\n📊 Performance Metrics:")
print(f"   Precision: {precision:.2%}")
print(f"   Recall:    {recall:.2%}")
print(f"   F1 Score:  {f1:.2%}")

# Prediction distribution
print(f"\n📊 Prediction Distribution:")
print(f"   Predicted CONSISTENT (1):    {(results_df['prediction'] == 1).sum()}")
print(f"   Predicted CONTRADICTORY (0): {(results_df['prediction'] == 0).sum()}")

print(f"\n📊 True Label Distribution:")
print(f"   True CONSISTENT (1):    {(results_df['true_label'] == 1).sum()}")
print(f"   True CONTRADICTORY (0): {(results_df['true_label'] == 0).sum()}")

# Analysis by book
print(f"\n📚 Accuracy by Book:")
for book in results_df['book_name'].unique():
    book_df = results_df[results_df['book_name'] == book]
    book_acc = (book_df['prediction'] == book_df['true_label']).mean()
    print(f"   {book}: {book_acc:.2%} ({len(book_df)} samples)")

# Sample errors
print(f"\n❌ Sample Errors (showing first 3):")
errors = results_df[results_df['prediction'] != results_df['true_label']].head(3)
for idx, row in errors.iterrows():
    print(f"\n   Sample ID: {row['id']}")
    print(f"   Book: {row['book_name']}")
    print(f"   Character: {row['character']}")
    print(f"   Predicted: {'CONSISTENT' if row['prediction'] == 1 else 'CONTRADICTORY'}")
    print(f"   Actual: {'CONSISTENT' if row['true_label'] == 1 else 'CONTRADICTORY'}")
    print(f"   Rationale: {row['rationale'][:150]}...")

# Check for bias
bias_score = results_df['prediction'].mean()
print(f"\n🎯 Model Bias:")
print(f"   Fraction predicting CONSISTENT: {bias_score:.2%}")
if bias_score < 0.3:
    print(f"   ⚠️ STRONG BIAS toward CONTRADICTORY")
elif bias_score > 0.7:
    print(f"   ⚠️ STRONG BIAS toward CONSISTENT")
else:
    print(f"   ✓ Relatively balanced predictions")

print("\n" + "=" * 80)