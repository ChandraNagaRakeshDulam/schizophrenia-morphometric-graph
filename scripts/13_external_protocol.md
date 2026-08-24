# External validation protocol

Do not choose any modeling decision using the external dataset.

1. Finalize on the development dataset:
   - FreeSurfer version
   - DK68 feature definitions
   - MSN construction
   - MIND construction
   - graph sparsification rule
   - model family and hyperparameters
   - preprocessing/scaling policy
2. Process external raw T1 with the exact same FreeSurfer container.
3. Apply the exact same QC rubric.
4. Build ROI/MSN/MIND subject representations independently.
5. Align ROI names only; never use external labels to choose transformations.
6. Fit train-derived scalers on development subjects only and apply them to external subjects.
7. Train the final classifier on all accepted development subjects.
8. Evaluate exactly once on the external cohort.
9. Report ROC-AUC, PR-AUC, balanced accuracy/sensitivity/specificity as appropriate, and subject-level bootstrap confidence intervals.
10. Compare ROI/edge attribution rankings between development and external cohorts without selecting top-k using external performance.

For leave-one-site-out, the entire pipeline that learns cross-subject parameters must be refit using the training sites only.
