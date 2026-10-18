## Summary

<!-- What does this change do, and why? -->

## Type of change

- [ ] Model / training change
- [ ] Inference or app change
- [ ] Bug fix
- [ ] Documentation
- [ ] Refactor

## Checklist

- [ ] `ruff check src tests app.py` passes
- [ ] `ruff format src tests app.py` passes
- [ ] `pytest tests -q` passes
- [ ] No dataset, video or image files added (they belong in releases, see `.gitignore`)
- [ ] No file over 50 MB added
- [ ] No API keys or credentials committed
- [ ] Updated `README.md` if behaviour, usage or published metrics changed

## If you changed the model

- [ ] Re-ran `python -m src.evaluate` and pasted the new numbers into the README
- [ ] Replaced `models/best.pt` and confirmed the app still loads it