# Panning Shot Analysis — Project Handoff

## Project Goal

Extract physical information (camera exposure trajectory + F1 car velocity) from a single
panning photograph, using a sharp reference image of the background taken from a nearby
but non-identical camera pose.

This is framed as a **structured inverse problem**: the blur in the image is not a
degradation to remove, but a signal to decode. The blur pattern encodes the camera's
rigid-body motion during exposure, modulated by scene depth.

---

## What Exists in the Codebase (Current State)

The pipeline currently runs up to and including power spectrum analysis:

1. **EXIF extraction** — reads exposure time, focal length, sensor size from image metadata
2. **Metric depth estimation** — uses known physical size of the F1 car as a reference
   object to estimate metric depth to the car via apparent size in the image
  -- actually not present right now, but can be easily figured out using the exposure time and focal length and known wheelbase length for f1 machines - 3.40 meters
3. **Blur direction estimation** — Sobel gradient analysis on local patches to estimate
   the dominant blur direction per patch
4. **Blur kernel length estimation** — power spectrum analysis on patches, assuming an
   initial isotropic variance and uniform box blur kernel, to estimate kernel length

**Known limitations of the current pipeline:**
- Patch selection is manual — no automatic identifiability check
- Each patch is estimated independently — no global consistency enforcement
- Blur is too extreme in background regions to recover a meaningful sharp image
- The pipeline currently terminates with a car speed estimate, which works but is fragile

---

## What Has Changed — New Approach

### Core reframe

Instead of estimating blur kernels **independently per patch**, the new formulation is:

> Jointly estimate the **camera motion trajectory during exposure** that best explains
> all observed blur patterns simultaneously, enforcing global physical consistency.

The camera executes one rigid-body motion during one exposure. Every pixel's blur kernel
is a deterministic projection of that same trajectory onto that pixel's location,
modulated by depth. Estimating patches independently discards this global constraint.

### New asset: sharp reference image of the background

A sharp image of the background scene is available, taken from a **nearby but
non-identical camera pose**. This is the most important new input. It enables:

- **Reference-based kernel estimation**: for background patches, directly compare the
  registered sharp patch to the blurred patch — kernel estimation becomes much better
  conditioned (deconvolution with known reference, not blind deconvolution)
- **Additional depth signal**: parallax between the sharp reference and the blurry image,
  caused by the pose difference, gives depth information since patches at different
  depths warp differently under the homography

The non-identical pose is handled by registering the sharp reference to the blurry
image via feature matching (SIFT or SuperPoint) and a depth-aware warp.

---

## New Pipeline Architecture

### Stage 1 — Register sharp reference to blurry image

- Feature matching between sharp background image and blurry image
  (SIFT/SuperPoint; background regions in blurry image retain perpendicular-to-blur
  edge structure sufficient for matching)
- Estimate homography or depth-aware warp (use existing EXIF depth estimate for car
  as anchor)
- Output: sharp reference warped into the coordinate frame of the blurry image

### Stage 2 — Dense blur kernel estimation from registered reference

- For each background patch: estimate blur kernel by comparing the registered sharp
  patch to the corresponding blurry patch
- This is now a well-conditioned problem — kernel is the function that maps known
  sharp signal to observed blurry signal
- Automatic patch selection: accept only patches with sufficient high-frequency content
  in the sharp reference (identifiability condition — flat patches carry no kernel info)
- Output: dense map of per-patch blur kernel estimates with confidence weights

### Stage 3 — Global trajectory fitting

- Parameterize camera motion during exposure as a low-dimensional curve
  (start with linear translation + rotation; 2-3 DOF given panning geometry)
- Each patch's expected blur is a deterministic function of:
  - The trajectory curve
  - The patch's depth (from EXIF depth estimate + parallax from registration)
  - The projection geometry
- Fit the trajectory to maximize consistency across all patches
- Objective: minimize weighted sum of residuals between predicted blur kernels and
  estimated blur kernels from Stage 2; downweight low-confidence patches
- Robustness: residual between re-blurred estimate and observed image should be
  structureless (white noise) — use this as a sanity check, not a pixel-wise loss

### Stage 4 — Car velocity estimation

- The car has no sharp reference, but its blur relative to the background encodes
  its velocity relative to the camera
- With the camera trajectory known from Stage 3, estimate the car's apparent motion
  from the *differential blur* between car and background
- Use known metric size of car + metric depth from Stage 1 to convert angular
  velocity to metric velocity
- Output: car speed in m/s (or km/h)

---

## Key Physical Constraints (do not discard these)

- EXIF exposure time bounds the total trajectory length
- Panning geometry constrains motion to ~1-2 DOF (not arbitrary 6-DOF)
- F1 car known metric dimensions provide absolute scale
- Camera focal length and sensor size from EXIF give pixel-to-angle calibration
- All background patches were generated by the **same** trajectory — global consistency
  is a hard constraint, not a soft prior

---

## Verification / Sanity Checks

- Re-blur the registered sharp reference with the estimated trajectory kernel and
  compare to the observed blurry image — residual should be structureless
- Blur direction from Stage 2 dense map should be spatially smooth (no abrupt
  direction flips between neighboring patches)
- Kernel length should correlate with depth in the expected direction (closer objects
  blur more for the same camera motion, unless they are the tracked subject)

---

## Implementation Notes for Claude Code

- The existing Sobel + power spectrum code can be reused in Stage 2 for patches where
  no sharp reference is available (e.g. the car itself), but the reference-based
  estimation should take priority for background patches
- Stage 1 registration: OpenCV `findHomography` + SIFT is a reasonable starting point;
  SuperPoint/SuperGlue if SIFT fails due to blur
- Stage 3 optimization: scipy `minimize` with L-BFGS-B is sufficient for the low-DOF
  trajectory parameterization; no need for a neural approach here
- Confidence weighting in Stage 3: use gradient energy of the sharp reference patch
  as the weight (high-texture patches → high confidence)
- Keep the physical units consistent throughout: work in radians and meters, convert
  only at output

---

## Open Questions / Known Hard Parts

- Can feature matching succeed between the sharp reference and blurry image?
  Linear blur streaks preserve perpendicular edge structure — test this first before
  committing to the registration approach
- The car region has no sharp reference — the differential blur estimation in Stage 4
  is the weakest link and may need a separate treatment
- Patch identifiability: need a principled threshold for "enough high-frequency content"
  — start with gradient energy in the sharp reference as a proxy

---

## Research Framing (context for decisions)

This project is a course project for a computational imaging course (SNU 4190.762)
and a potential seed for a research direction combining **3D vision** and
**computational photography**. The goal is not just a working speed estimator but a
demonstration that blur is a *structured signal to decode* rather than a degradation
to remove. Decisions about method should favor physical interpretability over
black-box accuracy.
