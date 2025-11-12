import subprocess
import re

results = []
for run in range(5):
    print(f"\n{'='*60}")
    print(f"RUN {run + 1}/5")
    print('='*60)
    
    result = subprocess.run(
        ['cargo', 'run', '--release', '--bin', 'test_adaptive_regression'],
        capture_output=True,
        cwd=r'C:\Users\asus\OneDrive - Value Score Business Solutions LLP\Desktop\PkBoost genesis rust'
    )
    
    output = result.stdout.decode('utf-8', errors='ignore') if result.stdout else ''
    
    # Extract key metrics
    entropy_match = re.search(r'Error Entropy: ([\d.]+)', output)
    drift_type_match = re.search(r'Drift Type: (\w+)', output)
    accepted_match = re.search(r'ACCEPTED: ([\d.]+) → ([\d.]+) \(([+-][\d.]+)%\)', output)
    rollback_match = re.search(r'ROLLBACK', output)
    final_rmse_matches = re.findall(r'Obs \d+: RMSE=([\d.]+)', output)
    meta_count_match = re.search(r'Metamorphoses triggered: (\d+)', output)
    
    run_data = {
        'run': run + 1,
        'entropy': float(entropy_match.group(1)) if entropy_match else None,
        'drift_type': drift_type_match.group(1) if drift_type_match else None,
        'rollback': rollback_match is not None,
        'meta_count': int(meta_count_match.group(1)) if meta_count_match else 0,
    }
    
    if accepted_match:
        run_data['before'] = float(accepted_match.group(1))
        run_data['after'] = float(accepted_match.group(2))
        run_data['improvement'] = float(accepted_match.group(3))
    
    if final_rmse_matches:
        run_data['final_rmse'] = float(final_rmse_matches[-1])
    
    results.append(run_data)
    
    print(f"Entropy: {run_data.get('entropy', 'N/A')}")
    print(f"Drift Type: {run_data.get('drift_type', 'N/A')}")
    print(f"Metamorphoses: {run_data['meta_count']}")
    if 'improvement' in run_data:
        print(f"Improvement: {run_data['improvement']:+.1f}%")
    print(f"Final RMSE: {run_data.get('final_rmse', 'N/A')}")

print(f"\n{'='*60}")
print("SUMMARY ACROSS 5 RUNS")
print('='*60)

entropies = [r['entropy'] for r in results if r['entropy']]
improvements = [r['improvement'] for r in results if 'improvement' in r]
final_rmses = [r['final_rmse'] for r in results if 'final_rmse' in r]
meta_counts = [r['meta_count'] for r in results]

print(f"Entropy: {sum(entropies)/len(entropies):.3f} ± {max(entropies)-min(entropies):.3f}")
print(f"Improvement: {sum(improvements)/len(improvements):+.1f}% ± {max(improvements)-min(improvements):.1f}%")
print(f"Final RMSE: {sum(final_rmses)/len(final_rmses):.3f} ± {max(final_rmses)-min(final_rmses):.3f}")
print(f"Success rate: {sum(meta_counts)}/{len(results)} ({sum(meta_counts)/len(results)*100:.0f}%)")
