//! Respect the container's CPU quota, rather than its visible host CPU count.
use std::fs;

pub fn quota_cores(text: &str) -> Option<f64> {
    let mut words = text.split_whitespace();
    let quota: f64 = words.next()?.parse().ok()?;
    let period: f64 = words.next()?.parse().ok()?;
    (quota > 0.0 && period > 0.0).then_some(quota / period)
}

pub fn available_workers() -> usize {
    let visible = std::thread::available_parallelism().map_or(1, usize::from);
    let mut cores = visible as f64;
    let mut quotas = Vec::new();
    if let Ok(text) = fs::read_to_string("/sys/fs/cgroup/cpu.max") {
        quotas.push(text);
    }
    for root in ["/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"] {
        if let (Ok(quota), Ok(period)) = (
            fs::read_to_string(format!("{root}/cpu.cfs_quota_us")),
            fs::read_to_string(format!("{root}/cpu.cfs_period_us")),
        ) {
            quotas.push(format!("{} {}", quota.trim(), period.trim()));
        }
    }
    for quota in quotas {
        if let Some(limit) = quota_cores(&quota) {
            cores = cores.min(limit);
        }
    }
    (cores.floor() as usize).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_fractional_and_unlimited_quotas() {
        assert_eq!(quota_cores("765000 100000"), Some(7.65));
        assert_eq!(quota_cores("25000 100000"), Some(0.25));
        assert_eq!(quota_cores("max 100000"), None);
        assert_eq!(quota_cores("-1 100000"), None);
        assert_eq!(quota_cores("100 0"), None);
    }
}
