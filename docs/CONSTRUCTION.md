# Construction of PReMind-Bench

> This document and `premind/construction/` describe the base pipeline for scenes, memories and requests. Final
> benchmark statistics are in the dataset card and the paper; the interdependency (conflict) cases and later
> refinements are described in the paper's appendix.

One principle guides construction: every reference answer must be supported by the visible scene and, when needed,
by the correct personal memory. The pipeline has five stages. Code lives in `premind/construction/`, and the prompt
templates appear in the paper's appendix.

## 1. Scene images
- **Sources.** We start from real multi-view captures of ten scenes from Mip-NeRF 360 and Zip-NeRF. Camera
  geometry comes from structure from motion (COLMAP).
- **Insertion.** An image model inserts one person, pet or object using a masked edit. Physical scale follows the
  camera projection, or a fully visible, independently sized reference at a compatible depth.
- **Checks.** Edits are rejected if they alter protected pixels, the viewpoint, fixed objects or scene detail.
  Semantic review checks identity and visible risk.
- **Targets.** Each image is generated with one or two of the user's concepts as inputs, and these inputs are the
  item's targets. Images generated without concept inputs are excluded.

## 2. Personal memory
- Each user store combines target concepts, 3–4 same-type look-alikes and other concepts, for 10–12 in total.
- A local VLM describes each concept's appearance and fine-grained type.
- The decisive attribute is chosen first. The target and each look-alike then receive distinct, lasting facts about
  the same action, so an answer requires identifying the instance. No fact is copied between concepts.
- Memory text is written in the third person and is scene-independent. People and pets may have names, and objects
  use natural handles. All reference images go through the same preprocessing.

## 3. Requests and reference answers
- **Authoring.** GPT-6-Sol sees the query image and memories and writes the requests, facts, reference answers,
  citations and labels.
- **Row types.**
  - G0 needs only the image.
  - G1-visual needs an instance fact.
  - G1+G2-visual also needs a general safety or health rule.
- **Request wording.** Requests omit names, distinguishing attributes and decisive values. For a memory-dependent
  request, replacing the target's fact with a look-alike's must change the supported answer. Image-only requests
  use the same advice-seeking register as personal requests.
- **Target-absent rows.** An absent target supports a withhold row only when the missing fact is decision-critical.
- **Acute hazards.** An acute visible hazard must be addressed first in every response derived from that image.

## 4. Checks and review
- **Code checks** cover fact consistency, query leakage, support patterns, citations and naming. Failed components
  receive a targeted repair and are checked again.
- **Independent review.** A primary reviewer and a local reviewer inspect the rendered evidence without seeing the
  author's rationale, and an adjudicator resolves disagreements. Review tests visual support, memory necessity,
  look-alike contrasts and safety.
- **Repairs.** A text-only repair cannot establish a new visual claim, so repaired rows are re-checked against their
  images. Rows that still fail are removed.

## 5. Benchmark assembly
- SigLIP2 supplies eight retrieved entries in a seeded display order that is independent of the labels.
  Target membership determines whether the required memory is available. Paired variants remove the target before
  retrieval.
- Generic calm and panic-style requests come from one shared pool and are assigned independently of the labels.
  This pool includes safe false-alarm controls whose answers are calm and grounded in the scene.
- Rows are rendered in the retrieved, GT-retrieved and target-drop conditions. Leakage checks keep related rows
  together during splitting (see [BENCHMARK.md](BENCHMARK.md)).

## Shortcut controls
- Calm and panic wordings are shared across labels. Safe implicit answers are grounded in the scene, and calls on
  these controls are neutral.
- Image-only requests were rewritten into the same normative register as personal requests, and lure items were
  added. Lexical cues still remain, and request-only probes measure them (see the paper's appendix on shortcut and
  leakage controls).
- Request-only baselines expose the topic cues that remain.

## Quality limits
- Image-aware review is necessary. A stratified audit found unsupported visual claims in 13 of 35 text-only
  additions, 5 of 35 text-only repairs and 2 of 30 image-authored rows.
- The local and primary judges agreed poorly (κ = 0.03–0.13), so local review does not substitute for image-grounded
  verification.
- A population-wide quality estimate is not yet available.

## Licensing and provenance
The following is the paper's wording:

- The Mip-NeRF 360 and Zip-NeRF captures are used, and their edited frames redistributed, with the authors' permission (J. T. Barron, e-mail of 2026-10-14). Their Apache-2.0 licenses cover the code.
- Concepts come from public personalization datasets (YoLLaVA, MyVLM, a curated subset of CustomConcept101 with
  quarantined images removed, DreamBooth, PODS) and from Unsplash person images.
- Per-image attribution for the Unsplash images is still to be added.
- Construction inputs and code are hash-pinned, outputs are write-once, and exclusions remain auditable.

The dataset card lists the source and redistribution status of each asset class.

## Reproducibility of construction
- Stages that call paid image or language models are shipped as documented code. They cannot be re-run without
  API keys, and their outputs are pinned in the release.
- The post-processing chain that turns the pinned outputs into the final release (assembly, rewrites, patches,
  splitting) runs on CPU from `premind/construction/` (see [REPRODUCE.md](REPRODUCE.md)).
