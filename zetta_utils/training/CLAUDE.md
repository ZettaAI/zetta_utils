# Training

PyTorch Lightning-based training framework for deep learning workflows.

## Core Components
Data Loading Infrastructure: LayerDataset (PyTorch dataset wrapper around zetta_utils.layer.Layer components), JointDataset (supports horizontal simultaneous and vertical sequential dataset joining), build_collection_dataset (creates datasets from neuroglancer annotation collections)

Sample Indexers: SampleIndexer (base class: maps integer indices to layer-specific indices), VolumetricStridedIndexer (uniform sampling from 3D volumes with configurable stride), VolumetricNGLIndexer (samples from neuroglancer annotation points), RandomIndexer (randomizes sampling order with/without replacement), ChainIndexer (chains multiple indexers sequentially), LoopIndexer (loops over inner indexer to match desired sample count)

Lightning Training Framework: lightning_train (main training orchestrator supporting local/remote distributed training), ZettaDefaultTrainer (custom trainer with checkpointing, logging, and model tracing), remote training via Kubernetes with multi-node DDP support

Data Samplers: TorchDataLoader (registered PyTorch DataLoader), TorchRandomSampler (registered PyTorch RandomSampler), SamplerWrapper (DDP-compatible sampler wrapper)

## Builder Registrations
Key Registered Components: LayerDataset (main dataset class), JointDataset (multi-dataset combiner), lightning_train (training orchestrator), ZettaDefaultTrainer (default trainer), VolumetricStridedIndexer (volume sampling), RandomIndexer (randomized sampling), TorchDataLoader (PyTorch DataLoader wrapper)

## Usage Patterns
Training Specification Structure:
```cue
"@type": "lightning_train"
regime: {"@type": "BaseRegimenConfig"}
trainer: {"@type": "ZettaDefaultTrainer"}
train_dataloader: {"@type": "TorchDataLoader", dataset: {"@type": "LayerDataset"}}
val_dataloader: {}
```

Integration Points: PyTorch Lightning for training loops, Weights & Biases for experiment tracking, Kubernetes for distributed training, Neuroglancer for annotation-based sampling, CloudVolume layers for data access

## Development Guidelines
Dataset Creation: Use LayerDataset for layer-based data access, combine datasets with JointDataset for multi-modal training, choose appropriate indexer based on sampling strategy, consider memory usage for large datasets

Training Configuration: Use ZettaDefaultTrainer for standard training workflows, configure appropriate batch sizes for available memory, use distributed training for large datasets, monitor training with W&B integration

Indexing Strategy: Use VolumetricStridedIndexer for uniform volume sampling, use VolumetricNGLIndexer for annotation-based sampling, chain indexers for complex sampling patterns, use RandomIndexer for shuffled training

Performance Optimization: Configure appropriate number of DataLoader workers, use efficient data loading patterns, profile data loading bottlenecks, consider data caching for repeated access
