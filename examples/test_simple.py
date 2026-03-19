#!/usr/bin/env python3
"""Simple test to isolate the panic issue"""

import numpy as np

try:
    from pkboost import PKBoostClassifier
    print("✅ pkboost import successful")
    
    # Create very simple test data
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X dtype: {X.dtype}, y dtype: {y.dtype}")
    print(f"X contiguous: {X.flags.c_contiguous}")
    print(f"y contiguous: {y.flags.c_contiguous}")
    
    # Try to create model
    model = PKBoostClassifier.auto()
    print("✅ Model created")
    
    # Try to fit - this should panic
    print("About to fit...")
    model.fit(X, y, verbose=True)
    print("✅ Fit completed")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
