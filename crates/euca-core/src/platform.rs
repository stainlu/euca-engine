//! Platform detection utilities for optimal runtime configuration.

/// Returns the number of performance (P) cores on the current system.
///
/// On Apple Silicon (M1/M2/M3/M4), this returns only the high-performance
/// cores, excluding efficiency (E) cores. This value should be used to
/// configure thread pools (e.g. tokio worker threads) that are sensitive
/// to latency, since E-cores have lower single-thread performance.
///
/// On non-Apple platforms, returns the total logical CPU count as a
/// conservative default (all cores are assumed performance-class).
///
/// # Usage
/// ```ignore
/// tokio::runtime::Builder::new_multi_thread()
///     .worker_threads(euca_core::platform::performance_core_count())
///     .build()
/// ```
pub fn performance_core_count() -> usize {
    #[cfg(target_os = "macos")]
    {
        macos_performance_cores().unwrap_or_else(|| {
            let total = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            log::debug!("Could not detect Apple Silicon P-core count, using total cores: {total}");
            total
        })
    }

    #[cfg(not(target_os = "macos"))]
    {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }
}

/// Query macOS sysctl for the number of performance-level logical CPUs.
/// Returns `None` on non-Apple-Silicon Macs (Intel) or if the sysctl fails.
#[cfg(target_os = "macos")]
fn macos_performance_cores() -> Option<usize> {
    use std::process::Command;
    // hw.perflevel0.logicalcpu = number of P-core logical CPUs (Apple Silicon only)
    let output = Command::new("sysctl")
        .args(["-n", "hw.perflevel0.logicalcpu"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout);
    let count = s.trim().parse::<usize>().ok()?;
    if count > 0 {
        log::info!("Apple Silicon detected: {count} performance cores");
        Some(count)
    } else {
        None
    }
}
