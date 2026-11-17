### Generalization gap (positive = worse on target)

| source | target | arch | norm | f1_int | f1_ext | gap_f1 | roc_auc_int | roc_auc_ext | gap_roc_auc |
|---|---|---|---|---|---|---|---|---|---|
| A | C | densenet121 | imagenet | 0.919 | 0.581 | 0.337 | 0.972 | 0.778 | 0.195 |
| C | A | densenet121 | imagenet | 0.768 | 0.871 | -0.102 | 0.907 | 0.910 | -0.003 |
| C | A | resnet50 | imagenet | 0.741 | 0.861 | -0.120 | 0.897 | 0.927 | -0.030 |
