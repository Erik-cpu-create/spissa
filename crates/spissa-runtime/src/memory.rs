// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rama Erik Esprada

use crate::{Result, RuntimeError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryBudget {
    limit_bytes: usize,
    current_bytes: usize,
    peak_bytes: usize,
}

impl MemoryBudget {
    pub fn new(limit_bytes: usize) -> Self {
        Self {
            limit_bytes,
            current_bytes: 0,
            peak_bytes: 0,
        }
    }

    pub fn unbounded() -> Self {
        Self::new(usize::MAX)
    }

    pub fn limit_bytes(&self) -> usize {
        self.limit_bytes
    }

    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }

    pub fn reserve(&mut self, bytes: usize, label: impl Into<String>) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let label = label.into();
        let next = self.current_bytes.checked_add(bytes).ok_or_else(|| {
            RuntimeError::MemoryBudgetExceeded {
                requested: bytes,
                current: self.current_bytes,
                limit: self.limit_bytes,
                label: label.clone(),
            }
        })?;

        if next > self.limit_bytes {
            return Err(RuntimeError::MemoryBudgetExceeded {
                requested: bytes,
                current: self.current_bytes,
                limit: self.limit_bytes,
                label,
            });
        }

        self.current_bytes = next;
        self.peak_bytes = self.peak_bytes.max(self.current_bytes);
        Ok(())
    }

    pub fn release(&mut self, bytes: usize, label: impl Into<String>) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        if bytes > self.current_bytes {
            return Err(RuntimeError::MemoryBudgetUnderflow {
                released: bytes,
                current: self.current_bytes,
                label: label.into(),
            });
        }
        self.current_bytes -= bytes;
        Ok(())
    }

    pub fn reset_peak(&mut self) {
        self.peak_bytes = self.current_bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_current_and_peak_memory() {
        let mut budget = MemoryBudget::new(100);
        budget.reserve(40, "a").unwrap();
        budget.reserve(30, "b").unwrap();
        budget.release(50, "done").unwrap();

        assert_eq!(budget.current_bytes(), 20);
        assert_eq!(budget.peak_bytes(), 70);
    }

    #[test]
    fn rejects_over_budget_reservation() {
        let mut budget = MemoryBudget::new(64);
        budget.reserve(40, "first").unwrap();
        let err = budget.reserve(25, "second").unwrap_err();

        assert!(matches!(err, RuntimeError::MemoryBudgetExceeded { .. }));
        assert_eq!(budget.current_bytes(), 40);
        assert_eq!(budget.peak_bytes(), 40);
    }

    #[test]
    fn rejects_release_underflow() {
        let mut budget = MemoryBudget::new(64);
        budget.reserve(16, "first").unwrap();
        let err = budget.release(17, "bad release").unwrap_err();

        assert!(matches!(err, RuntimeError::MemoryBudgetUnderflow { .. }));
        assert_eq!(budget.current_bytes(), 16);
    }
}
