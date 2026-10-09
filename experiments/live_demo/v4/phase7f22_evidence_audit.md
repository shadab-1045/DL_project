# Phase 7F.22: Read-Only Audit of Phase 7F.21 Evidence

## 1. Sample Independence Verification
The Phase 7F.21 regression suite evaluated a total of 15 frames originating from 10 unique, unaligned source images.

**Genuine Frames (5 unique images)**
- `Angelina_Jolie/05.jpg` (`d6a69f1c...`)
- `Angelina_Jolie/06.jpg` (`b146c5b0...`)
- `Angelina_Jolie/07.jpg` (`c58d0710...`)
- `Angelina_Jolie/08.jpg` (`3255c2b1...`)
- `Angelina_Jolie/09.jpg` (`c0764080...`)

**Different-Person Frames (5 unique images)**
- `Alejandro_Toledo/00.jpg` (`950abfd4...`)
- `Alejandro_Toledo/01.jpg` (`2ba943ca...`)
- `Alejandro_Toledo/02.jpg` (`ade605b1...`)
- `Alejandro_Toledo/03.jpg` (`8b26a026...`)
- `Alejandro_Toledo/04.jpg` (`a95fdee7...`)

**Impersonation Frames (0 new unique images)**
- Reused the exact 5 `Alejandro_Toledo` source images and applied the `Native InSwapper` in-memory attack to them.

*Both the old in-memory pipeline and the new parity-adapter pipeline received the exact same aligned pixel arrays for every frame. The Genuine sample `05.jpg` was selected purely programmatically via glob-sorting (`[5:10]`), not singled out post hoc.*

## 2. A/B Comparison Verification
The telemetry correctly records the divergence and convergence between the old visual path and the new adapter path.

**Genuine Pipeline Behavior:**
- `05.jpg`: Old P(Synth)=`0.7409` -> New P(Synth)=`0.1604`. P(Id_Match)=`0.9999`. UI Transitioned from `SUSPECTED_IMPERSONATION` to `VERIFIED` due to the adapter.
- `06.jpg` to `09.jpg`: Old P(Synth) ranged `0.9951-0.9992`. New P(Synth) ranged `0.9230-0.9976`. P(Id_Match) universally `1.0000`. UI securely mapped to `SUSPECTED_IMPERSONATION`.

**Different-Person Pipeline Behavior:**
- All 5 frames yielded negative Model A cosine similarities (e.g. `-0.0659` to `0.0185`), successfully triggering the rigid rejection gate. V4 was correctly not invoked (recorded as `null`/`0.0`). UI securely mapped to `UNKNOWN`.

**Impersonation Pipeline Behavior:**
- All 5 frames successfully fooled Model A (Cosine `~0.81`). 
- Old P(Synth) ranged `0.9998-1.0000`. New P(Synth) ranged `0.9976-1.0000`.
- P(Id_Match) universally `1.0000`. Fusion Class `2`. UI mapped securely to `SUSPECTED_IMPERSONATION`.

## 3. Integrity Verification
- **Model A Threshold**: Verified to remain exactly `0.244529`.
- **Checkpoint Hashes**: The SHA-256 hash for `best_fusion_model.pt` (`f2647012bc2863d988e65f980310cac9b60a0beb8c3e068f38b564f4383be2f5`) remained immutable before and after the suite.
- **Held-Out Test Set**: Confirmed completely untouched during Phase 7F.19-21.

## 4. Findings and Claims Summary

| Finding Type | Observation / Claim | Audit Verdict |
|--------------|---------------------|---------------|
| **Verified Finding** | The missing JPEG compression shortcut caused `05.jpg` to fail. | The adapter mathematically re-injected this artifact and perfectly corrected the specific frame's prediction from `0.74` to `0.16`. |
| **Verified Finding** | The adapter preserves Native InSwapper Impersonation detection. | True synthetic frames maintained `P(Synth) ~1.0` and successfully triggered `SUSPECTED_IMPERSONATION`. |
| **Supported Hypothesis** | The frozen V4 visual branch fails on Out-Of-Distribution (OOD) LFW frames. | Supported. `06.jpg`-`09.jpg` maintained extreme P(Synth) scores despite the JPEG artifact correction, proving the dataset-shift vulnerability extends beyond the JPEG shortcut. |
| **Unsupported Claim (Requires Correction)** | The OOD failure is caused specifically by "heavy noise/blur/lighting anomalies native to unconstrained LFW internet images." | **Unsupported.** These factors were never independently controlled or measured during Phase 7F.21. The only directly measured and controlled variable was JPEG parity. Specific attribution to blur or lighting is technically speculative and must be retracted from formal conclusions. |
