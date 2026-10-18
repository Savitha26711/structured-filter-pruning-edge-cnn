# structured-filter-pruning-edge-cnn


# Structured Filter Pruning and Weight Sparsification for Edge CNNs

## Project Overview
This project studies CNN model compression for edge devices
using a baseline CNN and a compact CNN architecture.

## Dataset
CIFAR-10 image classification dataset.

## Technologies
- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Google Colab

## Methodology
1. Load and preprocess CIFAR-10.
2. Train the baseline CNN.
3. Create a compact CNN with fewer channels.
4. Transfer compatible weights from the baseline model.
5. Fine-tune and evaluate the compact model.
6. Compare accuracy, parameter count, model size and latency.

## Files
- `main.py` - Project implementation
- `benchmark_results.csv` - Performance comparison
- `experiment_summary.csv` - Experiment summary
- `baseline_training_log.csv` - Baseline training log
- `compact_training_log.csv` - Compact model training log
- PNG files - Performance comparison graphs

## Execution
Run the Python implementation in Google Colab with GPU
enabled for faster training.

## Note
The current implementation demonstrates channel-width reduction,
compatible weight transfer and fine-tuning. Separate unstructured
weight sparsification is not yet implemented.
