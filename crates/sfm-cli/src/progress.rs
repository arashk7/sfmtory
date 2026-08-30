//! Progress reporting for the long parallel stages.
//!
//! `feature` and `match` are both a single `par_iter().collect()` over work
//! that can run for hours - 965 twelve-megapixel images take over three of
//! them - and until now both printed nothing at all between "Found N images"
//! and the final summary. There was no way to tell a slow run from a hung one,
//! or to find out that a run would need another two hours before committing to
//! waiting for it. That is the whole reason this exists.
//!
//! Output goes to stderr, because it is status rather than result, and as
//! discrete lines rather than a `\r`-updated one: the GUI streams these
//! straight into its log pane, where a carriage return would render as one
//! unreadable smear.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

pub struct Progress {
    label: &'static str,
    total: usize,
    done: AtomicUsize,
    started: Instant,
    /// Milliseconds since `started` when a line was last printed.
    last_ms: AtomicU64,
    /// Print every this many items, so a long run produces ~100 lines rather
    /// than one per item.
    step: usize,
}

/// Never print more often than this, however small the step - a fast stage
/// should not bury its own summary.
const MIN_INTERVAL_MS: u64 = 1000;

impl Progress {
    pub fn new(label: &'static str, total: usize) -> Self {
        let p = Progress {
            label,
            total,
            done: AtomicUsize::new(0),
            started: Instant::now(),
            last_ms: AtomicU64::new(0),
            step: (total / 100).max(1),
        };
        if total > 0 {
            eprintln!("{label}: 0/{total} starting");
        }
        p
    }

    /// Records one completed item, printing a line when one is due.
    pub fn tick(&self) {
        let done = self.done.fetch_add(1, Ordering::Relaxed) + 1;
        let last = done == self.total;
        if !last && !done.is_multiple_of(self.step) {
            return;
        }
        let elapsed_ms = self.started.elapsed().as_millis() as u64;
        if !last {
            // One thread claims the slot; the rest skip rather than queue up
            // behind a lock to print a line that is about to be superseded.
            let prev = self.last_ms.load(Ordering::Relaxed);
            if elapsed_ms.saturating_sub(prev) < MIN_INTERVAL_MS
                || self
                    .last_ms
                    .compare_exchange(prev, elapsed_ms, Ordering::Relaxed, Ordering::Relaxed)
                    .is_err()
            {
                return;
            }
        } else {
            self.last_ms.store(elapsed_ms, Ordering::Relaxed);
        }

        let elapsed = elapsed_ms as f64 / 1000.0;
        let frac = done as f64 / self.total.max(1) as f64;
        let rate = if elapsed > 0.0 {
            done as f64 / elapsed
        } else {
            0.0
        };
        let eta = if rate > 0.0 && done < self.total {
            format!("  eta {}", human_secs((self.total - done) as f64 / rate))
        } else {
            String::new()
        };
        eprintln!(
            "{}: {done}/{} {:>5.1}%  elapsed {}{}  ({})",
            self.label,
            self.total,
            frac * 100.0,
            human_secs(elapsed),
            eta,
            human_rate(rate),
        );
    }

    /// Total wall-clock time, for the caller's own summary line.
    pub fn elapsed_secs(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }
}

/// `4h07m`, `12m41s`, `9.3s` - two significant units, never more.
pub fn human_secs(s: f64) -> String {
    if !s.is_finite() || s < 0.0 {
        return "?".into();
    }
    let t = s.round() as u64;
    match t {
        0..=59 => format!("{s:.1}s"),
        60..=3599 => format!("{}m{:02}s", t / 60, t % 60),
        _ => format!("{}h{:02}m", t / 3600, (t % 3600) / 60),
    }
}

/// Rate in whichever direction reads better: items per second when fast,
/// seconds per item when slow. A stage doing one item every 13 seconds is far
/// clearer as "13.1 s/item" than as "0.1 items/s".
fn human_rate(per_sec: f64) -> String {
    if per_sec <= 0.0 || !per_sec.is_finite() {
        return "-".into();
    }
    if per_sec >= 1.0 {
        format!("{per_sec:.1}/s")
    } else {
        format!("{:.1} s each", 1.0 / per_sec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_read_as_two_units() {
        assert_eq!(human_secs(9.3), "9.3s");
        assert_eq!(human_secs(59.0), "59.0s");
        assert_eq!(human_secs(60.0), "1m00s");
        assert_eq!(human_secs(761.0), "12m41s");
        assert_eq!(human_secs(3599.0), "59m59s");
        assert_eq!(human_secs(3600.0), "1h00m");
        assert_eq!(human_secs(12683.0), "3h31m");
        assert_eq!(human_secs(f64::NAN), "?");
    }

    #[test]
    fn rate_flips_direction_around_one_per_second() {
        assert_eq!(human_rate(13.0), "13.0/s");
        assert_eq!(human_rate(1.0), "1.0/s");
        // The slow case is the one that matters: 13.1 s/image, not 0.1 images/s.
        assert_eq!(human_rate(1.0 / 13.1), "13.1 s each");
        assert_eq!(human_rate(0.0), "-");
    }

    #[test]
    fn every_item_is_counted_and_the_last_one_always_prints() {
        let p = Progress::new("test", 250);
        for _ in 0..250 {
            p.tick();
        }
        assert_eq!(p.done.load(Ordering::Relaxed), 250);
        // step is total/100, so the final tick lands on the exact total even
        // when it is not a multiple of the step.
        let p = Progress::new("test", 7);
        for _ in 0..7 {
            p.tick();
        }
        assert_eq!(p.done.load(Ordering::Relaxed), 7);
    }
}
