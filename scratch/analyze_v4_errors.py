import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_analysis():
    df = pd.read_csv('experiments/model_c_v4/test_predictions.csv')
    
    # 1. CM & Metrics
    print("--- 1. Confusion Matrix ---")
    cm = pd.crosstab(df['true_label'], df['predicted_class'])
    print(cm)
    
    # 2. Impersonation Errors
    imp_mask = df['true_label'] == 'impersonation'
    imp_err = df[imp_mask & ~df['correct']]
    
    print("\n--- 2. Impersonation Errors (FN) ---")
    print(f"Total: {len(imp_err)} missed impersonations")
    print("Generators:")
    print(imp_err['generator'].value_counts())
    print("\nPredicted Classes:")
    print(imp_err['predicted_class'].value_counts())
    
    # Check if they failed due to low P(Synth) or low P(Id)
    # Impersonation should have high P(Id) and high P(Synth)
    print("\nError P(Synth) stats vs Correct P(Synth) stats:")
    imp_corr = df[imp_mask & df['correct']]
    print(f"  Correct Imp P(Synth) Mean: {imp_corr['p_synthetic'].mean():.4f}")
    print(f"  Error Imp P(Synth) Mean: {imp_err['p_synthetic'].mean():.4f}")
    print(f"  Error Imp P(Id) Mean: {imp_err['p_identity_match'].mean():.4f}")
    
    # 3. Genuine Errors
    gen_mask = df['true_label'] == 'genuine'
    gen_err = df[gen_mask & ~df['correct']]
    gen_corr = df[gen_mask & df['correct']]
    print("\n--- 3. Genuine Errors ---")
    print(f"Total: {len(gen_err)} misclassified genuine")
    print(gen_err['predicted_class'].value_counts())
    print(f"  Correct Gen P(Id) Mean: {gen_corr['p_identity_match'].mean():.4f}")
    print(f"  Error Gen P(Id) Mean: {gen_err['p_identity_match'].mean():.4f}")
    print(f"  Correct Gen P(Synth) Mean: {gen_corr['p_synthetic'].mean():.4f}")
    print(f"  Error Gen P(Synth) Mean: {gen_err['p_synthetic'].mean():.4f}")
    
    # 4. Different Person Errors
    diff_mask = df['true_label'] == 'different_person'
    diff_err = df[diff_mask & ~df['correct']]
    diff_corr = df[diff_mask & df['correct']]
    print("\n--- 4. Different Person Errors ---")
    print(f"Total: {len(diff_err)} misclassified different person")
    if len(diff_err) > 0:
        print(f"  Correct Diff P(Id) Mean: {diff_corr['p_identity_match'].mean():.4f}")
        print(f"  Error Diff P(Id) Mean: {diff_err['p_identity_match'].mean():.4f}")
        print(f"  Correct Diff Raw Cosine Mean: {diff_corr['raw_cosine'].mean():.4f}")
        print(f"  Error Diff Raw Cosine Mean: {diff_err['raw_cosine'].mean():.4f}")
        
    # 5. 2D Decision Space
    print("\n--- 5. 2D Space Statistics ---")
    for cls in ['genuine', 'different_person', 'impersonation']:
        sub = df[df['true_label'] == cls]
        print(f"\n{cls.upper()}:")
        print(f"  P(Id_Match) - Mean: {sub['p_identity_match'].mean():.4f}, Median: {sub['p_identity_match'].median():.4f}, Std: {sub['p_identity_match'].std():.4f}, Min: {sub['p_identity_match'].min():.4f}, Max: {sub['p_identity_match'].max():.4f}")
        print(f"  P(Synth)    - Mean: {sub['p_synthetic'].mean():.4f}, Median: {sub['p_synthetic'].median():.4f}, Std: {sub['p_synthetic'].std():.4f}, Min: {sub['p_synthetic'].min():.4f}, Max: {sub['p_synthetic'].max():.4f}")
        
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='p_identity_match', y='p_synthetic', hue='true_label', style='correct', alpha=0.7)
    plt.axvline(x=0.5, color='gray', linestyle='--')
    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.title('V4 Test Predictions in 2D Feature Space')
    plt.savefig('experiments/model_c_v4/scatter_test.png')
    
    # 6. Hardest Samples
    # A simple metric for ambiguity is how close the point is to the decision boundaries.
    # We can measure distance to (0.5, 0.5) or just variance between probabilities (though we don't have the 3-class logits here).
    # Since we have only [P(Id), P(Synth)], ambiguity happens when P is near 0.5.
    df['ambiguity_score'] = -abs(df['p_identity_match'] - 0.5) - abs(df['p_synthetic'] - 0.5)
    hardest = df.sort_values(by='ambiguity_score', ascending=False).head(50)
    hardest.to_csv('experiments/model_c_v4/hardest_test_samples.csv', index=False)
    print("\n--- 6. Saved 50 hardest samples to hardest_test_samples.csv ---")

if __name__ == "__main__":
    run_analysis()
