# Testing Guidelines

## General Principles

- **No test classes** - Use plain functions, not class-based test organization
- **Minimal mocking** - Use real backends with temp directories whenever possible; only mock when absolutely necessary
- **Use mocker fixture** - When mocking is required, use the `mocker` fixture (pytest-mock), never unittest mocks

## File Organization

```
tests/unit/layer/volumetric/<module>/
├── test_<component>.py    # One file per component
```

## Naming Conventions

- Test files: `test_<component>.py`
- Test functions: `test_<feature>_<scenario>`

## Fixtures

- Function-scoped (default)
- Use `tempfile.TemporaryDirectory()` for temp dirs
- Explicit cleanup via context managers or yield pattern

## Assertions

- `pytest.approx()` for float comparisons
- `np.testing.assert_array_equal()` for array comparisons
- `pytest.raises(ExceptionType, match="pattern")` for exception testing
