import optunahub
import logging

logging.basicConfig(level=logging.INFO)

try:
    print("Loading AutoSampler from OptunaHub...")
    module = optunahub.load_module(package="samplers/auto_sampler")
    AutoSampler = module.AutoSampler
    print(f"Successfully loaded AutoSampler: {AutoSampler}")
    
    # Instantiate it
    sampler = AutoSampler()
    print(f"Successfully instantiated AutoSampler: {sampler}")
except Exception as e:
    print(f"Failed to load AutoSampler: {e}")
    import traceback
    traceback.print_exc()
