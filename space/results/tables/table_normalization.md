### Normalization comparison (external)

| source | target | arch | f1_imagenet | f1_minmax | roc_auc_imagenet | roc_auc_minmax | roc_auc_delta(minmax-imagenet) | f1_delta(minmax-imagenet) |
|---|---|---|---|---|---|---|---|---|
| A | C | densenet121 | 0.581 | 0.586 | 0.778 | 0.795 | 0.017 | 0.005 |
| A | C | resnet50 | nan | 0.561 | nan | 0.755 | nan | nan |
| C | A | densenet121 | 0.871 | 0.861 | 0.91 | 0.891 | -0.019 | -0.009 |
| C | A | resnet50 | 0.861 | 0.802 | 0.927 | 0.817 | -0.11 | -0.059 |
