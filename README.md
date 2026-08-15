# README

**EntropyML** is a compact machine‑learning framework that presents neural models
through thermodynamic concepts such as entropy, equilibrium, and free‑energy‑like
objectives. The framework provides small, readable implementations intended for
experimentation and conceptual study.

EntropyML includes autoencoders, variational autoencoders, and a thermodynamic
variant of the VAE. These models are implemented in plain C# and are designed to
be structurally simple and transparent.

The project was previously named **ThermoML**. The name was changed to avoid
collision with the existing NIST ThermoML standard.

## Components

### Autoencoder (AE)

A minimal autoencoder implementation for reconstruction experiments.

### Variational Autoencoder (VAE)

A standard VAE implementation with clear latent‑space structure.

### Thermodynamic VAE (TVAE)

A variant of the VAE that introduces equilibrium and entropy‑potential concepts
for interpreting latent behavior.

### Neural Network (NN)

A simple feedforward network used in examples.

### Data Utilities

Small utilities for synthetic and example datasets.

### Examples

Runnable examples demonstrating AE, VAE, TVAE, NN, and data utilities.

## Repository Structure

```
EntropyML-Dev/
    background/
    doc/
    solution/
        EntropyML/
            EntropyML.AE/
            EntropyML.Data/
            EntropyML.NN/
            EntropyML.VAE/
            EntropyML.TVAE/
            Examples/
```

## Documentation

Documentation is currently being migrated to the EntropyML identity. The
following documents have been renamed:

- EntropyML_API_Sheet.md
- EntropyML_FolderStructure.md
- EntropyML_HMD.md
- EntropyML_Landing.md
- EntropyML_QuickStart.md
- EntropyML_ReleaseBundle.md
- EntropyML_SpecLite.md
- EntropyML_Terminology.md
- EntropyML_Versioning.md

These documents will be updated after the implementation surface stabilizes.

## Running Examples

Examples can be executed directly using the .NET command line:

```
dotnet run
```

Each example directory contains its own project file.

## License

MIT License. See the LICENSE file for details.
