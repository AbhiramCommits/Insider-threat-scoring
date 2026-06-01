# Feature Catalog

| # | Feature Name | Category | Description | Risk Rationale |
|---|-------------|----------|-------------|----------------|
| 1 | days_active | Temporal | Number of distinct days the user appeared in logs | Inactive users may be dormant accounts; unusually low activity can signal account takeover |
| 2 | after_hours_ratio | Temporal | Proportion of days with a login outside 07:00–19:00 | After-hours activity is a strong indicator of data exfiltration attempts when fewer monitors are present |
| 3 | weekend_ratio | Temporal | Proportion of activity days that fall on Saturday / Sunday | Weekend logins are rare for most roles; elevated ratios correlate with unauthorised off-hours access |
| 4 | login_hour_mean | Temporal | Mean hour of the day the user logs in | A shift in mean login time relative to peer baseline may indicate compromised credentials |
| 5 | login_hour_variance | Temporal | Variance of login hour across days | High variance suggests irregular access patterns; low variance with an off-peak mean is equally suspicious |
| 6 | login_hour_std | Time Distribution | Standard deviation of daily login hour | Complements variance; large spreads indicate unpredictable behaviour |
| 7 | login_hour_min | Time Distribution | Earliest recorded login hour | Extremely early logins (e.g. 02:00–04:00) are anomalous for most office-based roles |
| 8 | login_hour_max | Time Distribution | Latest recorded login hour | Consistent late-night activity is a classic insider-threat precursor |
| 9 | login_hour_median | Time Distribution | Median login hour over the observation window | Robust measure of central tendency; large gaps between median and mean reveal skewed schedules |
| 10 | login_hour_skew | Time Distribution | Skewness of the login-hour distribution | Positive skew indicates a tail of late-night sessions; negative skew points to early-morning outliers |
| 11 | morning_ratio | Time Bucket | Proportion of logins between 07:00–11:59 | Low morning ratio combined with high evening ratio suggests schedule inversion |
| 12 | afternoon_ratio | Time Bucket | Proportion of logins between 12:00–18:59 | Normal baseline; sudden drops can indicate avoidance of office hours |
| 13 | evening_ratio | Time Bucket | Proportion of logins between 19:00–23:59 | Elevated evening ratios are a well-known exfiltration signature in CERT datasets |
| 14 | night_ratio | Time Bucket | Proportion of logins between 00:00–06:59 | Very rare for legitimate users; persistent night access is high-risk |
| 15 | usb_event_total | Activity – USB | Total USB insertion/removal events across all days | USB mass-storage is the most common physical exfiltration vector |
| 16 | usb_event_spike | Activity – USB | Binary flag: any day where USB events exceed 2 z-scores above the user's own mean | Even one extreme spike is indicative of bulk data transfer |
| 17 | usb_event_mean | Activity – USB | Mean daily USB events | Establishes the user's normal peripheral usage; large totals with low means hide single-day bursts |
| 18 | usb_event_max | Activity – USB | Maximum USB events on a single day | A single high-volume day is more suspicious than consistently moderate usage |
| 19 | usb_days_ratio | Activity – USB | Proportion of days with at least one USB event | Frequent USB use may be benign for some roles, but sudden onset is a red flag |
| 20 | email_volume_spike | Activity – Email | Binary flag: any day where sent-email count exceeds 2 z-scores | Bulk emailing is a data-exfiltration technique (sending documents to personal accounts) |
| 21 | external_email_ratio_mean | Activity – Email | Mean proportion of emails sent to external domains | Insiders leaking data typically communicate with external recipients at elevated rates |
| 22 | email_volume_total | Activity – Email | Total emails sent over the observation window | Very high totals paired with high external ratios indicate systematic data egress |
| 23 | email_volume_mean | Activity – Email | Mean daily email sent count | Contextualises total volume; a user with high mean and high external ratio is high-risk |
| 24 | email_volume_max | Activity – Email | Maximum emails sent on a single day | Single-day bursts can reflect a "smash-and-grab" data theft pattern |
| 25 | email_volume_std | Activity – Email | Standard deviation of daily email sent count | High variability suggests intermittent bulk-send behaviour |
| 26 | email_external_ratio_max | Activity – Email | Maximum external-email ratio observed on any day | A single day with predominantly external recipients is a strong exfiltration signal |
| 27 | email_external_ratio_std | Activity – Email | Standard deviation of the external-email ratio | Volatile external ratios indicate sporadic data sharing with outside parties |
| 28 | file_access_anomaly_score | Activity – File | Maximum z-score of daily file-access counts relative to the user's own history | Peaks in file access map to bulk-download / staging behaviour prior to exfiltration |
| 29 | file_access_total | Activity – File | Total files accessed across all days | Unusually high totals relative to peers suggest data-hoarding |
| 30 | file_access_mean | Activity – File | Mean daily file-access count | Normalises the total; a high mean with a high spike flags persistent bulk access |
| 31 | file_access_max | Activity – File | Maximum files accessed on a single day | Single-day mass access is a hallmark of insider data collection |
| 32 | file_access_std | Activity – File | Standard deviation of daily file-access count | Large fluctuations hint at intermittent bulk access rather than steady work |
| 33 | file_access_cv | Activity – File | Coefficient of variation of file-access count (std / mean) | High CV with high max values suggests concentrated download events |
| 34 | failed_login_total | Activity – Access | Total failed login attempts across all days | Repeated failed logins indicate credential guessing, privilege-escalation probes, or account takeover |
| 35 | failed_login_mean | Activity – Access | Mean daily failed login count | Persistent low-level failures can be reconnaissance for lateral movement |
| 36 | failed_login_max | Activity – Access | Maximum failed logins on a single day | A brute-force burst is highly indicative of malicious intent |
| 37 | failed_login_ratio | Activity – Access | Proportion of days with at least one failed login | Frequent failures suggest systematic probing rather than occasional typos |
| 38 | vpn_usage_ratio | Activity – VPN | Proportion of days where VPN was used | VPN can mask true location; elevated VPN use with after-hours activity increases risk |
| 39 | vpn_usage_total | Activity – VPN | Total number of VPN-connected days | High VPN totals combined with high external email ratios suggest obfuscated exfiltration |
| 40 | login_hour_deviation_count | Deviation (7d rolling) | Number of days where login_hour deviated >2 std from its 7-day rolling mean | Detects schedule anomalies that a simple aggregate would smooth out |
| 41 | login_hour_max_deviation | Deviation (7d rolling) | Maximum z-score deviation of login_hour from 7-day rolling mean | Quantifies the worst schedule disruption; large values signal abrupt routine changes |
| 42 | login_hour_deviation_ratio | Deviation (7d rolling) | Proportion of days with login_hour deviation >2 std | High ratios imply persistently erratic access times |
| 43 | email_sent_count_deviation_count | Deviation (7d rolling) | Number of days where email_sent_count deviated >2 std from its 7-day rolling mean | Sporadic email bursts that would otherwise blend into the mean are surfaced |
| 44 | email_sent_count_max_deviation | Deviation (7d rolling) | Maximum z-score deviation of email_sent_count from 7-day rolling mean | Captures the single most anomalous email day in context of recent behaviour |
| 45 | email_sent_count_deviation_ratio | Deviation (7d rolling) | Proportion of days with email_sent_count deviation >2 std | Consistent email anomalies indicate sustained data egress |
| 46 | file_access_count_deviation_count | Deviation (7d rolling) | Number of days where file_access_count deviated >2 std from its 7-day rolling mean | Flags days where file access spiked relative to the user's own recent pattern |
| 47 | file_access_count_max_deviation | Deviation (7d rolling) | Maximum z-score deviation of file_access_count from 7-day rolling mean | The worst file-access spike contextualised against the preceding week |
| 48 | file_access_count_deviation_ratio | Deviation (7d rolling) | Proportion of days with file_access_count deviation >2 std | Repeated file-access anomalies suggest ongoing data staging |
| 49 | usb_after_hours_total | Interaction | Sum of USB events that occurred during after-hours periods | USB activity outside business hours multiplies the risk of both behaviours |
| 50 | email_after_hours_total | Interaction | Sum of emails sent during after-hours periods | Bulk emailing late at night is a strong exfiltration signal |
| 51 | high_risk_day_count | Composite Risk | Number of days where ≥3 anomalous indicators fire simultaneously | Multi-signal days are rare for benign users and common during threat events |
| 52 | avg_risk_flags | Composite Risk | Mean number of anomalous indicators per day | Continuous risk score; higher values correlate with more pervasive threat behaviour |
| 53 | label | Target | Ground-truth insider-threat label (1 = threat) | Supervised learning target; used for training and evaluation only |
