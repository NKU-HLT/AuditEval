import os
from typing import List

from swift.llm import BaseArguments, InferRequest, PtEngine, get_template

os.environ['IMAGE_MAX_TOKEN_NUM'] = '1024'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '16'


original_caption = "Keys jingle as a car attempts to start."
target_caption = "Keys jingle as a car attempts to start with a man speaking."

prompt = """You are a professional audio editing quality evaluator.

You will be given:
(1) the original audio,
(2) the edited audio,
(3) a textual description of the original audio (original_caption),
(4) a textual description of the target edited audio (target_caption).

The semantic difference between original_caption and target_caption implicitly defines the intended audio edit.
Your task is to evaluate how well the edited audio reflects this semantic change,
while maintaining high audio quality and preserving unedited content.

Evaluate the edited audio from three perspectives: Quality, Relevance, and Faithfulness.
Each score should be an integer from 1 to 5 (higher is better).
Output only three integer scores, without explanations.

1. Quality (1–5):
Evaluate the perceptual quality of the edited audio compared to the original,
including intelligibility, fluency, naturalness, and background noise.

- 5: As good as or better than the original, with clear, fluent, and natural audio.
- 4: Slight degradation in at most one aspect, but overall quality remains acceptable.
- 3: Noticeable degradation in multiple aspects or clear degradation in one aspect, but still understandable.
- 2: Significant artifacts that seriously harm naturalness or intelligibility.
- 1: Severely damaged or unintelligible audio.

2. Relevance (1–5):
Evaluate how well the edited audio matches the semantic content described in target_caption,
relative to original_caption.
Focus only on the parts that are different between original_caption and target_caption.

- 5: The edited audio accurately and naturally reflects the target_caption, with correct and complete semantic changes.
- 4: The main semantic change is correctly reflected, with only minor imperfections.
- 3: The general semantic intent is correct, but details are partially misaligned or awkward.
- 2: The semantic change is loosely reflected, with clear errors or unnatural execution.
- 1: The edited audio does not match the target_caption.

3. Faithfulness (1–5):
Evaluate how well the audio content that should remain unchanged (according to the captions)
is preserved in the edited audio, in terms of rhythm, order, loudness, prosody, and style.
Do not penalize the semantic changes required by target_caption.

- 5: Unchanged content is perfectly preserved and indistinguishable from the original.
- 4: Mostly preserved, with only minor variation.
- 3: Noticeable changes in rhythm or prosody, but overall style remains similar.
- 2: Major changes in unedited content with limited resemblance.
- 1: Unedited content is heavily altered or missing.

Original Caption:
{original_caption}

Target Caption:
{target_caption}
"""

test_audios = ["./examples/sys1_add_ori_--8puiAGLhs_30000_40000.wav", "./examples/sys3_add_tar_--8puiAGLhs_30000_40000.wav"]
infer_request = InferRequest(
    messages=[{
        'role':'user',
        'content': prompt,
    }],
    audios=test_audios)
adapter_path = './AuditEval-llm/ckpt/auditeval-llm'
args = BaseArguments.from_pretrained(adapter_path)

engine = PtEngine(
    args.model,
    adapters=[adapter_path],
    task_type='seq_cls',
    num_labels=args.num_labels,
    problem_type=args.problem_type)
template = get_template(args.template, engine.processor, args.system, use_chat_template=args.use_chat_template)
engine.default_template = template

resp_list = engine.infer([infer_request])
response: List[int] = resp_list[0].choices[0].message.content

print("========================================== AuditEval-llm ==========================================")
# print(f'response: {response}')
print(f"[INFO] Original text: {original_caption}")
print(f"[INFO] Target text:   {target_caption}")

print(f"[INFO] Original audio: {test_audios[0]}")
print(f"[INFO] Target(edited) audio: {test_audios[1]}\n")

print("[RESULT] Quality Score:      {:.4f}".format(response[0]))
print("[RESULT] Relevance Score:    {:.4f}".format(response[1]))
print("[RESULT] Faithfulness Score: {:.4f}".format(response[2]))
print("========================================== AuditEval-llm ==========================================\n")