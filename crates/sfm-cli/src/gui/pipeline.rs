//! Runs pipeline stages from the GUI.
//!
//! Stages are launched as child processes of this same binary rather than
//! called in-process. That keeps the viewer responsive during a long
//! reconstruction, isolates it from a stage that panics, and matches how the
//! stages are meant to be used - each one an independent process communicating
//! through the project directory.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub struct RunState {
    pub log: Arc<Mutex<Vec<String>>>,
    pub running: Arc<AtomicBool>,
    /// Set when a run finishes, so the viewer knows to reload the model.
    pub finished: Arc<AtomicBool>,
    pub last_stage: Arc<Mutex<String>>,
}

impl RunState {
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    pub fn take_finished(&self) -> bool {
        self.finished.swap(false, Ordering::Relaxed)
    }

    pub fn push(&self, line: impl Into<String>) {
        if let Ok(mut l) = self.log.lock() {
            l.push(line.into());
            // Bound the buffer: a long run can emit a lot, and the viewer only
            // ever shows the tail.
            let len = l.len();
            if len > 4000 {
                l.drain(0..len - 3000);
            }
        }
    }

    /// Spawns `sfmtory <args>` against `project`, streaming its output.
    pub fn spawn(&self, project: &Path, args: Vec<String>) {
        if self.is_running() {
            return;
        }
        let exe = match std::env::current_exe() {
            Ok(e) => e,
            Err(e) => {
                self.push(format!("cannot locate the sfmtory binary: {e}"));
                return;
            }
        };
        self.running.store(true, Ordering::Relaxed);
        if let Ok(mut s) = self.last_stage.lock() {
            *s = args.first().cloned().unwrap_or_default();
        }
        self.push(format!(
            "$ sfmtory {} --project {}",
            args.join(" "),
            project.display()
        ));

        let log = self.log.clone();
        let running = self.running.clone();
        let finished = self.finished.clone();
        let project = project.to_path_buf();
        std::thread::spawn(move || {
            let mut cmd = Command::new(exe);
            cmd.args(&args)
                .arg("--project")
                .arg(&project)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let push = |line: String| {
                if let Ok(mut l) = log.lock() {
                    l.push(line);
                }
            };
            match cmd.spawn() {
                Ok(mut child) => {
                    // Drain stderr on its own thread; stages write progress
                    // there and a full pipe would otherwise block the child.
                    let err_log = log.clone();
                    let stderr = child.stderr.take();
                    let err_thread = std::thread::spawn(move || {
                        if let Some(e) = stderr {
                            for line in BufReader::new(e).lines().map_while(Result::ok) {
                                if let Ok(mut l) = err_log.lock() {
                                    l.push(line);
                                }
                            }
                        }
                    });
                    if let Some(out) = child.stdout.take() {
                        for line in BufReader::new(out).lines().map_while(Result::ok) {
                            push(line);
                        }
                    }
                    let _ = err_thread.join();
                    match child.wait() {
                        Ok(st) if st.success() => push("-- done --".into()),
                        Ok(st) => push(format!("-- failed ({st}) --")),
                        Err(e) => push(format!("-- failed: {e} --")),
                    }
                }
                Err(e) => push(format!("failed to start: {e}")),
            }
            finished.store(true, Ordering::Relaxed);
            running.store(false, Ordering::Relaxed);
        });
    }
}
