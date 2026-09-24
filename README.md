# Age-related reorganization of locus coeruleus–cortical functional connectivity gradients

Code, source data and stimuli for the manuscript (Emotion and Aging Neuroscience Lab, Kavli Institute for Systems Neuroscience, NTNU).
Corresponding author: Maryam Ziaei maryam.ziaei@ntnu.no

**Preprint:** [bioRxiv, doi:10.64898/2026.02.05.704005](https://www.biorxiv.org/content/10.64898/2026.02.05.704005v2)

## Quick start: reproduce the main results

```
pip install pandas numpy scipy statsmodels openpyxl matplotlib seaborn
python "Analysis Scripts/wrapper/Wrapper.py"
```

`Wrapper.py` runs the statistics reported in the paper on the released master sheet (140 participants: 72 younger, 68 older adults), prints them, and writes one CSV per analysis to `replication_output/` at the repository root. It uses only the code and data in this repository: age-group comparisons of gradient dispersion (negative and neutral movie, gradient range, leave-one-out reliability as covariates), brain–behaviour correlations and age-moderation models (10,000 permutations, seed 42), the pontine-tegmentum control seed, the valence/arousal ratings, and summaries of the saved reliability results.

## Contents

| Folder | What it holds |
|---|---|
| `Analysis Scripts/` | MATLAB live scripts: `Gradient.mlx` (LC–cortical connectivity gradients), `individual_to_group.mlx` (individual to group-level gradients), `dispersion_main_F.mlx` (gradient dispersion analysis), `splithalf and kfold.mlx` (split-half and k-fold reliability). Python: `age_dispersion_behaviour_functions.py` (group comparisons of dispersion with covariate adjustment, brain–behaviour correlations with permutation testing, FDR correction, age-moderation analysis), `surface_plot.py` (projects gradient maps onto the fsaverage surface). Jupyter notebooks: `PT seed age difference.ipynb` (pontine-tegmentum control seed – age-group differences in gradient dispersion and the seed × age test; uses `group_comparison()` from `age_dispersion_behaviour_functions.py`, run from the repository root), `valence arousal behaviour.ipynb` (mixed ANOVA and paired tests of self-reported valence and arousal ratings across baseline, neutral and negative conditions by age group). `wrapper/Wrapper.py` runs all of the Python analyses in one go (see Quick start). |
| `Source Data/Disperion_source_data/` | `All_in_1_MasterSheet.xlsx` – one sheet with every participant-level value used in the statistics: demographics, valence/arousal ratings, dispersion and range measures for both conditions and hemispheres, head motion, cortical thickness, behavioural indices, pontine-tegmentum dispersion and leave-one-out reliability. `Source_data_LC_Cortex_Gradient.xlsx` – the same values split into sheets by measure (values underlying the main figures). |
| `Source Data/Sensitivity_Analyses_source_data/` | `LOO_LC_Cortex_Gradient.xlsx` (leave-one-out reliability per participant), plus `inter_individual_var/`, `k_fold/` and `Split_half/` with `.mat` results per condition (negative/neutral) and LC hemisphere (left/right). |
| `Movie stimuli/` | The two naturalistic movie clips used during fMRI (stored with Git LFS; install [Git LFS](https://git-lfs.com) before cloning). |

## Notes

- In the master sheet, `Sex` is numeric (1 = female, 0 = male) and `Gender` is the same information as text (F/M). The Python functions take numeric covariates, so use `Sex` (as `Wrapper.py` does).
- Covariates: sex, head motion (mean framewise displacement of the same condition) and mean cortical thickness for the age comparisons; sex and head motion for the brain–behaviour models.
