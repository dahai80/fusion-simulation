---
doc_id: fusion-simulation/changelog
product: fusion-simulation
class: app
entry: service
doc_type: changelog
title: { zh: "fusion-simulation 变更日志", en: "fusion-simulation changelog" }
audience: engineer
status: draft
version: 0.1.11
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - CHANGELOG.md
---

# Changelog

All notable changes to fusion-simulation are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Adopt the fusion docs-system v1.1 contract (app x service): add `docs/product-page.md`, `docs/get-started.md`, `docs/api-reference.md`, `docs/config-reference.md`, `docs/architecture.md`, `docs/troubleshooting.md` with 12-key frontmatter, plus `ci/validate_docs.py` and a docs validation GitHub workflow.

### Changed

- Add docs-contract frontmatter to `README.md`, `README_CN.md`, `AGENTS.md`, `CLAUDE.md`.

## [0.1.11] - 2026-10-31

### Fixed

- Formula: pin url + sha256 to the v0.1.11 release assets.

## [0.1.10] - 2026-10-29

### Fixed

- Relax the fusion-core git pin to `>=0.9.0` to fix workspace editable overwrite (issue #16).
- Sync `__version__` to 0.1.10 so `fusion-sim version` no longer misreports v0.1.9 after a Homebrew install.

## [0.1.9] - 2026-08-24

### Changed

- Hardening release for the v0.1.8 audit (22 P0-P2 blockers): removed fake progress and fabricated training data, moved LLM inference out of the blocking frame loop (async + circuit breaker / backoff / rate limit), real `pybullet.saveState` / `restoreState` snapshots with an LRU cap, global + per-episode RNG determinism, proto bytes encoding for camera frames, long-run leak fixes. See README "Reliability & Determinism (v0.1.9)".

## [0.1.8] - 2026-08-24

### Added

- v0.1.8 feature baseline covered by the audit referenced above.
