# Worst 30% participant error analysis (K=5)

Development split: meta-validation only. Locked meta-test was not accessed.

- Participants: 697
- Worst-tail threshold: participant mean MAE >= 9.969 mmHg
- Tail participants: 210
- Support-to-query BP change uses only the first K calibration events as the support anchor.
- PPG standard deviation is a simple amplitude proxy, not a validated signal-quality index.

## Group comparison

| group | mean_mae | sbp_mae | dbp_mae | target_sbp_mean | target_sbp_std | target_dbp_mean | target_dbp_std | event_index_mean | event_index_max | ppg_f_std_mean | abs_delta_support_sbp_mean | abs_delta_support_dbp_mean | outside_support_sbp_rate | outside_support_dbp_rate | support_sbp_std | support_dbp_std | age_clean_mean | age_valid_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| remaining_70pct_subjects | 6.6558 | 8.4181 | 4.8936 | 116.7543 | 11.8205 | 61.6717 | 6.5349 | 71.4743 | 136.9487 | 0.2747 | 13.9841 | 7.4652 | 0.5657 | 0.5534 | 9.1795 | 5.2255 | 61.1753 | 0.9959 |
| worst_30pct_subjects | 13.7142 | 17.6906 | 9.7378 | 122.2040 | 15.2379 | 64.9386 | 8.8441 | 99.0833 | 192.1667 | 0.2765 | 19.0247 | 10.5462 | 0.6403 | 0.6227 | 9.9273 | 5.6568 | 64.1014 | 0.9857 |

## Source composition

| source | participants | mean_mae | sbp_mae | dbp_mae | high_error_rate |
| --- | --- | --- | --- | --- | --- |
| MIMIC | 316 | 9.6331 | 12.3554 | 6.9107 | 0.3829 |
| VitalDB | 381 | 8.0770 | 10.2633 | 5.8906 | 0.2336 |

Observed query error defines the tail, so this is diagnostic/oracle analysis only; it is not a deployable rejection rule.
