from snake_ai.main import Trainer
import torch.multiprocessing as mp
import time

if __name__ == "__main__":
    print("Test: Starting Batched Inference Verification")
    
    # Force spawn for Windows compatibility (crucial!)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Create trainer with minimal settings
    # We need to hack the constants in main.py or just accept defaults
    # For speed, we will monkeypatch constants if possible, or just run and wait.
    import snake_ai.main as main
    main.GAMES_PER_GEN = 2
    main.SIMULATIONS = 5
    main.BATCH_SIZE = 4
    
    trainer = Trainer(run_dir="experiments/test_batching", start_gen=0, num_workers=2)
    
    print("Test: Running 1 generation...")
    avg_score = trainer.train_generation()
    
    print(f"Test: Generation finished. Score: {avg_score}")
    print("Test: SUCCESS")
