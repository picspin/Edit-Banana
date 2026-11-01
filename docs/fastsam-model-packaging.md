# FastSAM-s weights: packaging (not committed to git)

## What is NOT in the repo, and why
- The weights file `FastSAM-s.pt` (22.7 MB) is an upstream
  ultralytics asset, not a file this repo owns. It is NOT
  committed to git history in this branch, by design: a binary
  in git history bloats every clone and cannot be removed cleanly
  later. The integration round (2026-11-02) deliberately kept the
  weights out and used an env-var path instead.
- Current reference copy (round 3, scratch only):
  `/Users/hilbert/work/editbanana-round3/FastSAM-s.pt`
  Source (from the round-3 download log):
  `https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt`
  Verify a copy against this SHA-256 before trusting it
  (recorded 2026-11-02 from the round-3 scratch copy):
  c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844
  Pin the release tag too; if upstream re-releases the file the
  hash can change even at the same filename.

## Three packaging options
1. **Local path via env var (current, dev/test only)**
   - Set `EDITBANANA_FASTSAM_MODEL=/path/to/FastSAM-s.pt`.
   - Pros: no repo change, no download at runtime, easy to swap
     weights for experiments.
   - Cons: a fresh checkout on another machine has no weights;
     the scratch path above is machine-specific and will break
     elsewhere. This is why it is NOT the production default.
2. **Bundle with the app (recommended for production)**
   - Ship `FastSAM-s.pt` next to the app (e.g. an app `models/`
     directory) and point `EDITBANANA_FASTSAM_MODEL` at it in the
     app's launch config, or set the module default at package
     time to the bundled path.
   - Pros: offline, reproducible, version-pinned with the app.
   - Cons: +22.7 MB in the app bundle; the packager must record
     the upstream release tag and SHA-256 at bundle time.
3. **Download on first use (not recommended without pinning)**
   - On first FastSAM call, download from the pinned release URL
     above into a user cache directory if the file is absent.
   - Pros: small bundle, no manual step.
   - Cons: needs network on first use, needs a pinned URL +
     SHA-256 check (an unpinned download can silently change
     behaviour), and needs a clear failure mode when offline.
     Do not implement an unpinned auto-download.

## Behaviour when weights are missing
- `detect_fastsam_masks()` raises `FileNotFoundError`.
- `DetectionContourExtractor.extract()` catches it, records
  `fastsam_available=False` and `fastsam_error`, and returns
  contours + YOLO results unchanged. The pipeline never crashes
  for a missing FastSAM file, and it never reports FastSAM
  results it did not actually compute.
- `EDITBANANA_FASTSAM_MODEL=""` (empty string) disables FastSAM
  entirely, same as the YOLO `""` convention.

## What this doc does NOT claim
- FastSAM does not give separate arrow heads (head-only masks
  were 0 on all round-3 scratch images). "Arrow instance" means
  shaft+head as one mask, never head separation.
- Dense-diagram elongated masks are not arrows without a
  gap-spanning check (see modules/fastsam_detection.py).
- Single-chain diagonal arrows are a known miss for the current
  horizontal gap-spanning classifier (round-4: 0/3). Packaging
  does not fix that; it is a separate detection round.
