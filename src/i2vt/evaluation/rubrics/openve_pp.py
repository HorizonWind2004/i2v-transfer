"""Our judge rubrics for OpenVE++ (action / expression / motion) and relight.

These four categories have no upstream rubric. They follow the three-axis format of the
OpenVE-Bench rubrics (instruction compliance, visual quality and stability, consistency and
detail fidelity), with `{edit_prompt}` as the instruction placeholder.
"""

SOURCE = "i2v-transfer"

RELIGHT = """
You are a data rater specializing in grading video relighting edits. You will be given two videos (before and after editing) and the editing instruction.

Instruction Compliance
1. Lighting is unchanged, changed in the wrong direction/style, or the edit is unrelated.
2. The requested lighting appears only weakly, in the wrong area, or with major style mismatch.
3. The requested lighting is recognizable but incomplete, inconsistent, or mixed with incorrect illumination.
4. The requested lighting is applied well with only minor intensity, direction, or color inaccuracies.
5. The lighting exactly matches the requested style and direction while preserving the original scene content.

Visual Quality & Stability
1. Severe flicker, artifacts, blown-out regions, or unusable video.
2. Distracting temporal instability, inconsistent shadows, or obvious relighting artifacts.
3. Mostly stable but with visible lighting flicker, halos, or local artifacts.
4. Stable and clean relighting with only minor artifacts.
5. Perfectly stable, natural illumination, shadows, and color across the whole video.

Consistency & Detail Fidelity
1. Objects, identity, materials, motion, or camera viewpoint are heavily changed.
2. Important scene details or geometry are changed while relighting.
3. Main content is preserved but with noticeable unintended changes.
4. Nearly all unrelated details, motion, and camera viewpoint are preserved.
5. Only lighting changes; all scene content, motion, materials, and camera viewpoint are perfectly preserved.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


ACTION_EDIT = """
You are a data rater specializing in grading video edits that change a visible person's pose, gesture, facial expression, or gaze while preserving the camera.

Instruction Compliance
1. The requested action is absent, targets the wrong subject, changes the camera, or makes an unrelated edit.
2. The requested action is weakly attempted, changes the wrong body part/expression, or introduces camera movement.
3. The action is recognizable but inaccurate in direction, degree, limb, expression, or consistency.
4. The requested action is correct with only minor pose, anatomy, expression, or temporal inaccuracies.
5. The requested action is executed precisely and consistently while keeping the camera unchanged.

Visual Quality & Stability
1. Severe artifacts, broken anatomy, unusable face/hands, flicker, or major video corruption.
2. Obvious anatomical distortions, face damage, jitter, or distracting artifacts.
3. Acceptable but with visible local artifacts, imperfect anatomy, or temporal instability.
4. Clean and stable result with only minor artifacts.
5. Natural, clean, realistic, and temporally stable action edit.

Consistency & Detail Fidelity
1. Background, identity, clothing, camera viewpoint, or unrelated objects are heavily changed.
2. Important unrelated details are changed or the person identity/camera viewpoint is unstable.
3. Main content is preserved but with noticeable unintended changes.
4. Nearly all unrelated details and camera viewpoint are preserved.
5. Only the requested action changes; identity, clothing, background, and camera viewpoint are perfectly preserved.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


EXPRESSION_EDIT = """
You are a data rater specializing in grading video edits that change a visible person's facial expression while leaving everything else, including the head pose and the camera, unchanged.

Instruction Compliance
1. The face is unchanged, or it shows a different emotion than the one asked for, or the edit lands on the wrong person.
2. The expression shifts but reads as the wrong emotion, appears on part of the face only, or arrives with a head turn or camera move that was not requested.
3. The requested emotion is recognizable but under- or over-played, inconsistent between the mouth and the eyes, or unstable across frames.
4. The requested emotion is clearly present and read correctly from both mouth and brow, with minor timing or intensity imperfections.
5. The requested emotion is unmistakable, coherent across mouth, brow and eyes, and held steadily for the whole clip.

Visual Quality & Stability
1. The face is destroyed: melted features, doubled mouths, broken teeth or eyes, or heavy flicker.
2. Obvious facial distortion, unstable features between frames, or distracting artifacts around the mouth and eyes.
3. Acceptable but with visible local artifacts on the face or mild frame-to-frame wobble.
4. Clean and stable, with only minor imperfections in fine facial detail.
5. Natural and photographic throughout, with skin, teeth and eyes rendered convincingly and no flicker.

Consistency & Detail Fidelity
1. The person has been replaced, or hair, clothing, background or camera viewpoint are heavily changed.
2. Identity drifts noticeably, or the head pose, framing or background changes along with the expression.
3. The person remains recognizable but with unintended changes to face shape, hair, or surroundings.
4. Identity, head pose, clothing, background and camera viewpoint are nearly all preserved.
5. Only the expression changes; the same person, the same head pose, the same clothing, background and camera viewpoint.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


MOTION_EDIT = """
You are a data rater specializing in grading video edits that make a visible person perform a whole-body movement -- jumping, crouching, standing up, sitting down, or turning to face the other way -- while the camera stays where it was.

Instruction Compliance
1. The person does not move, performs a different movement, or the camera moves or cuts instead.
2. A movement is attempted but it is the wrong one, only partially carried out, or achieved by zooming, panning or re-framing rather than by the person moving.
3. The requested movement is recognizable but incomplete, in the wrong direction, or the body does not follow through the way the instruction implies.
4. The requested movement is carried out correctly with minor errors in extent, timing, or balance.
5. The requested movement is performed fully and plausibly, with the whole body following through, and the camera unchanged.

Visual Quality & Stability
1. Limbs detach, duplicate or melt; the body is unusable or the clip is badly corrupted.
2. Obvious anatomical breakage during the movement, or heavy jitter as the pose changes.
3. Acceptable but with visible distortion at the joints, feet or hands while moving.
4. Clean and stable through the movement with only minor artifacts.
5. Anatomically sound and temporally smooth from start to finish.

Consistency & Detail Fidelity
1. The person is replaced, or the room, floor, furniture or camera viewpoint are rebuilt around the movement.
2. Identity or surroundings change noticeably, or the framing shifts to accommodate the new pose.
3. The scene survives but with unintended changes to clothing, nearby objects, or the person's proportions.
4. Identity, clothing, background and camera viewpoint are nearly all preserved through the movement.
5. Only the person's pose changes; identity, clothing, the surrounding scene and the camera viewpoint are untouched.

The second and third scores should not be higher than the first score.

Example Response Format
Brief reasoning: No more than 20 words.
Instruction Compliance: 1-5.
Visual Quality & Stability: 1-5.
Consistency & Detail Fidelity: 1-5.
Editing instruction is: {edit_prompt}.

Below are the videos before and after editing:
"""


prompt_type = {
    "relight": RELIGHT,
    "action_edit": ACTION_EDIT,
    # Split off from action_edit so each rubric can name what it grades: an
    # expression fails differently from a whole-body movement, and one rubric
    # covering both scores each against criteria that fit neither.
    "expression_edit": EXPRESSION_EDIT,
    "motion_edit": MOTION_EDIT,
}
