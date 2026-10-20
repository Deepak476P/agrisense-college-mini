# Stepwise project development

The Git history groups this implementation into six logical milestones.
Intermediate commits are learning milestones; use the latest `main` commit for the complete running app.
These are real commits created during this implementation, not backdated work.

| Step | Change | Why it belongs here |
|---|---|---|
| 1 | Project setup, dependencies, VS Code configuration, source datasets, illustrations and provenance | Establish a reproducible environment and identify the supplied data |
| 2 | Dataset checks, schemas, model pipelines, cross-validation, three saved models and measured metrics | Build and evaluate the tabular machine-learning foundation |
| 3 | Nutrient comparisons, quantity/economics arithmetic, rainfall scenarios and leaf image processing | Add transparent calculations and the fourth feature |
| 4 | Integrated Streamlit interface | Make all four modules usable through forms, charts, validation and result displays |
| 5 | Five executed notebooks and detailed documentation | Explain what was done, why, how the code works and how to interpret results |
| 6 | Automated checks, verification utility and recorded execution evidence | Verify complete workflows and record reproducible evidence |

## Follow the implementation

1. Read `notebooks/00_project_guide.ipynb` for the architecture and full feature map.
2. Study `src/data.py`: paths, input columns, target columns and validation.
3. Study `src/ml.py`: split, preprocessing, candidate comparison, final fitting and prediction.
4. Read notebooks 01 and 02 alongside the classifier code.
5. Study `src/advisory.py` and notebook 03 for equations and assumptions.
6. Study `src/vision.py` and notebook 04 for colour masks and image limitations.
7. Open `app.py` to see how the functions connect to the interface.
8. Run `tools/verify.py` and read `reports/verification.json`.

## See the steps in Git

```powershell
git log --oneline --reverse
git show --stat <commit-id>
git diff <earlier-commit-id> <later-commit-id> -- src
```

On GitHub, open the repository's commit history to view each milestone and its changed files.
You do not need to check out older commits to study their diffs.

## Suggested student presentation

Explain the problem and source data, show the pipeline and evaluation, demonstrate all four features,
then discuss limitations. Use the notebook's recorded results. Do not describe synthetic accuracy as
field accuracy, colour-change ratios as disease confidence, or assumed prices as current market prices.
