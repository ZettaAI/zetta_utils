# Builder

Registry and factory system for dynamic object creation from declarative specifications.

## Core Registry System (registry.py)
Registry Components: Global Registry (REGISTRY dict mapping string names to lists of RegistryEntry objects), RegistryEntry (dataclass containing fn, allow_partial, allow_parallel, and version_spec fields), Version Support (uses packaging.specifiers.SpecifierSet for semantic version matching), Entry Resolution (get_matching_entry() finds unique registry entries by name and version)

## Building System (building.py)
Multi-stage Building: Object Graph (ObjectToBeBuilt represents nodes with functions, kwargs, and parent relationships), Dependency Stages (parses specs into dependency stages for parallel/sequential execution), Execution Engine (_execute_build_stages() processes stages with optional parallel execution via ProcessPoolExecutor), BuilderPartial (delayed execution wrapper for partial function application)

Core Building Functions: build() (main entry point accepting spec dict/list or CUE file path), _parse_stages() (converts nested specs into execution stages), _build_object() (executes individual function calls with error context), Special builders for lists, tuples, and dicts

## Registration Patterns
Decorator Usage:
```python
@builder.register("build_cv_layer", versions=">=0.4")
def build_cv_layer(path: str, ...):
    # Implementation
```

Registration Parameters: allow_partial (enable @mode: "partial" for delayed execution), allow_parallel (control parallel execution eligibility), versions (semantic version specification string)

Built-in Registrations: Lambda string parsing (@type: "lambda"), NumPy function auto-registration (all public numpy functions with np. prefix), eval-based lambda execution with security length limits

## Usage Patterns
CUE Specification Format:
```cue
{
  "@type": "build_cv_layer"
  "@version": "0.4.0"
  path: "gs://bucket/path"
  // ... other parameters
}
```

Special Keys: @type (registry lookup name), @version (version specification), @mode ("partial" or "regular" - default)

Nested Building: Specs can contain other specs, building dependency graphs automatically resolved, automatic parallelization of independent build stages, error context preservation through build chain

## Development Guidelines
Function Registration: Use descriptive registry names that reflect purpose, specify version constraints for breaking changes, set allow_partial=True for operations that benefit from delayed execution, set allow_parallel=False for operations with side effects

Spec Design: Use consistent parameter naming across related builders, provide sensible defaults to minimize required configuration, document expected parameter types and constraints, consider backward compatibility when updating builders

Error Handling: Builder preserves error context through build chain, use descriptive error messages that reference spec location, validate parameters early in builder functions, handle missing dependencies gracefully

Performance Considerations: Leverage parallel execution for independent operations, use @mode: "partial" for expensive operations that may not be needed, consider caching for expensive builder operations, profile build times for complex specifications
