"""Segmentation contracts for pet row strips.

Fast, hermetic, no generation mocks — these are the structural contracts the
hatch retry policy keys off, so they belong in the default suite. The
image-processing suite in ``tests/agent/test_pet_generate.py`` is opt-in behind
``HERMES_RUN_SLOW_PET_TESTS``, which CI does not set; contracts that guard a
paid-retry decision should not live only there.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from agent.pet.generate import atlas

SLOT = 208
HEIGHT = 208


def _strip_of(widths: list[int]) -> Image.Image:
    """One row strip with one opaque ellipse per slot, each *widths[i]* px wide."""
    img = Image.new("RGBA", (SLOT * len(widths), HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, width in enumerate(widths):
        cx = i * SLOT + SLOT // 2
        draw.ellipse((cx - width // 2, 34, cx + width // 2, 174), fill=(60, 80, 200, 255))
    return img


def _merged_pair_strip(count: int = 6) -> Image.Image:
    """Slots 3 and 4 drawn as ONE connected blob straddling the gutter."""
    img = _strip_of([140] * count)
    draw = ImageDraw.Draw(img)
    draw.ellipse((3 * SLOT - 140, 24, 5 * SLOT - 76, 184), fill=(200, 80, 80, 255))
    return img


def _two_poses_in_one_slot_strip(count: int = 6) -> Image.Image:
    """One slot holds two poses merged side by side — wide, not tall.

    This is the reported "split into two" frame: padded segmentation succeeds
    (the blob sits inside its slot with margins), so only frame validation can
    reject it.
    """
    return _strip_of([60] * 3 + [300] + [60] * (count - 4))


def _frame(width: int, height: int, *, opaque: bool = True) -> Image.Image:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if opaque:
        ImageDraw.Draw(img).rectangle((0, 0, width - 1, height - 1), fill=(60, 80, 200, 255))
    return img


def test_merged_poses_raise_the_structural_error_not_a_bare_valueerror():
    # Strict mode must fail with the STRUCTURAL error type: the orchestrator
    # keys off this class to skip the remaining strict (paid) retries instead of
    # substring-matching error text (#87739).
    with pytest.raises(atlas.UnsegmentableStripError):
        atlas.extract_strip_frames(_merged_pair_strip(), 6, method="components")


def test_strict_mode_surfaces_validation_failures_structurally_too():
    # Extraction can succeed while the ART is still unusable — a slot holding
    # two merged poses is one connected subject with margins, so padded slicing
    # accepts it and only validation rejects it. That is the same defect class
    # as an unsegmentable strip (the model drew two characters), so it must
    # reach the orchestrator as the same structural error, not a bare ValueError
    # that would trigger more paid strict re-rolls.
    with pytest.raises(atlas.UnsegmentableStripError):
        atlas.extract_strip_frames(_two_poses_in_one_slot_strip(), 6, method="components")


def test_auto_still_salvages_a_strip_strict_mode_rejects():
    # ``auto`` is the lenient path: the new exception must not change its
    # behavior. Uses a strip strict mode genuinely rejects, so this exercises
    # the salvage path rather than a strip both methods accept.
    strip = _merged_pair_strip()
    with pytest.raises(atlas.UnsegmentableStripError):
        atlas.extract_strip_frames(strip, 6, method="components")
    assert len(atlas.extract_strip_frames(strip, 6, method="auto")) == 6


def test_row_frames_collapsed_flags_slivers_and_passes_whole_poses():
    # A lenient salvage can "succeed" with thin fragments of a body that match
    # each other, so nothing marks them as outliers until compose rejects the
    # whole atlas. Compare the row's median width against what the character's
    # silhouette implies at that height.
    reference = (92, 145)
    assert atlas.row_frames_collapsed([_frame(92, 145) for _ in range(6)], reference) is None
    reason = atlas.row_frames_collapsed([_frame(32, 145) for _ in range(6)], reference)
    assert reason is not None and "sliver" in reason


def test_row_frames_collapsed_abstains_without_a_reference():
    # No identity anchor decoded (or an empty one) means no judgement: the gate
    # must not guess, and must not reject a row on a missing reference.
    frames = [_frame(32, 145) for _ in range(6)]
    assert atlas.row_frames_collapsed(frames, None) is None
    assert atlas.row_frames_collapsed(frames, (0, 0)) is None


def test_row_frames_collapsed_reports_a_row_with_no_art():
    assert atlas.row_frames_collapsed([_frame(92, 145, opaque=False) for _ in range(6)], (92, 145)) == "row has no visible frames"


def test_row_frames_collapsed_rejects_a_uniformly_shrunk_row():
    # A uniformly shrunk row (reference 92x145, frames ~40x60) walks past a
    # width-only test scaled by measured height — expected_w = med_h * ref_w /
    # ref_h is invariant under uniform shrink — yet compose rejects each cell
    # for being too short once ``validate_atlas`` runs, after every row has been
    # paid for. Rejecting on EITHER axis closes that gap.
    reference = (92, 145)
    reason = atlas.row_frames_collapsed([_frame(40, 60) for _ in range(6)], reference)
    assert reason is not None and "short" in reason


def test_unrelated_segmentation_valueerror_is_not_reclassified(monkeypatch):
    # A broad ``except ValueError`` around the whole slicing block turns ANY
    # ValueError into UnsegmentableStripError, which tells the orchestrator to
    # skip its remaining strict (paid) retries — even when the failure is not in
    # the art at all. Only the decision points may raise the structural type.
    def boom(*_args, **_kwargs):
        raise ValueError("unrelated failure deep in segmentation")

    monkeypatch.setattr(atlas, "_component_crops", boom)
    with pytest.raises(ValueError) as excinfo:
        atlas.extract_strip_frames(_strip_of([140] * 6), 6, method="components")
    assert not isinstance(excinfo.value, atlas.UnsegmentableStripError)


# ───────────────── one frame = exactly one whole character ─────────────────
# ``frame_defects`` is the ONE home for the invariant; these are the two
# morphologies the fleet audit found in the wild (side-by-side duplicates and a
# body split into stacked halves) plus the legitimate poses it must not reject.
# The reference is a single 72x100 body in every case.

FIGURE = (72, 100)


def _blobs(size, boxes):
    """RGBA canvas of *size* with one opaque ellipse per ``(left, top, right, bottom)``."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        draw.ellipse(box, fill=(60, 80, 200, 255) if i % 2 == 0 else (200, 80, 60, 255))
    return img


def test_frame_defects_accepts_a_single_figure():
    assert atlas.frame_defects(_blobs((72, 100), [(0, 0, 71, 99)]), FIGURE) == []


def test_frame_defects_rejects_side_by_side_duplicates():
    # Two similar-mass figures at different x. A bbox proxy merges them into one
    # (merely wide) box, and at ~2.2x the box stays under every 2.6x/3.0x
    # threshold in the pipeline — the live shy-ghost defect. Counting figures
    # does not merge.
    frame = _blobs((160, 100), [(2, 0, 73, 99), (86, 0, 157, 99)])
    reasons = atlas.frame_defects(frame, FIGURE)
    assert reasons and any("figures" in reason for reason in reasons)


def test_frame_defects_rejects_a_stacked_split_body():
    # One body split into two stacked halves at the same x (the cuddle-squish
    # morphology). Together the halves carry the mass of ONE body, so the area
    # ratio cannot see the defect and the bbox is unchanged — the eroded-core
    # count is the signal that survives.
    frame = _blobs((72, 100), [(0, 0, 71, 48), (0, 52, 71, 99)])
    reasons = atlas.frame_defects(frame, FIGURE)
    assert reasons and any("figures" in reason for reason in reasons)


def test_frame_defects_rejects_overlapping_figures_that_erode_into_one_core():
    # Two bodies overlapping enough to fuse after erosion. The core count then
    # under-reports, so the opaque-mass ratio is what catches it (~1.4x a single
    # body) — the two signals cover each other.
    frame = _blobs((124, 100), [(0, 0, 71, 99), (52, 0, 123, 99)])
    reasons = atlas.frame_defects(frame, FIGURE)
    assert reasons and any("opaque mass" in reason for reason in reasons)


def test_frame_defects_accepts_a_wide_pose():
    # Arms out: 1.6x wider than the reference and still ONE figure. Rejecting
    # this is the false positive the 2.6x/3.0x thresholds were raised to avoid;
    # the invariant must not reintroduce it.
    frame = _blobs((136, 100), [(0, 0, 71, 99), (58, 40, 115, 61), (20, 40, 77, 61)])
    assert atlas.frame_defects(frame, FIGURE) == []


def test_frame_defects_accepts_a_small_detached_lobe():
    # A cape/tail fragment is a legitimately disconnected island (the slicing
    # code comments rely on it). It erodes to a small core and must not be
    # counted as a second figure.
    frame = _blobs((104, 100), [(0, 0, 69, 99), (86, 30, 101, 69)])
    assert atlas.frame_defects(frame, FIGURE) == []


def test_frame_defects_counts_cores_without_a_reference():
    # The core count needs no identity anchor: with no reference the size/mass
    # signals abstain, but a doubled frame is still two figures.
    doubled = _blobs((160, 100), [(2, 0, 73, 99), (86, 0, 157, 99)])
    assert atlas.frame_defects(doubled, None) != []
    assert atlas.frame_defects(_blobs((72, 100), [(0, 0, 71, 99)]), None) == []


def test_frame_defects_flags_an_empty_frame():
    assert atlas.frame_defects(_frame(10, 10, opaque=False), FIGURE) == ["frame is empty"]


def test_validate_atlas_rejects_a_cell_holding_two_figures():
    # Post-compose wiring: compose must not accept what the pre-compose gate
    # rejects. One state's cell holds a doubled figure; the rest are single
    # bodies that define the atlas-wide median reference.
    def single():
        return _blobs((atlas.CELL_WIDTH, atlas.CELL_HEIGHT), [(60, 40, 131, 139)])

    states = {state: [single() for _ in range(count)] for state, _row, count in atlas.ROW_SPECS}
    assert atlas.validate_atlas(atlas.compose_atlas(states))["ok"]

    states["idle"][2] = _blobs((atlas.CELL_WIDTH, atlas.CELL_HEIGHT), [(10, 40, 81, 139), (94, 40, 165, 139)])
    result = atlas.validate_atlas(atlas.compose_atlas(states))
    assert not result["ok"]
    assert any("cell 2" in error and "figures" in error for error in result["errors"])

