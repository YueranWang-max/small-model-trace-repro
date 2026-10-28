# Submission metadata

Author: Yueran Wang

Affiliation: School of Artificial Intelligence, Beijing Normal University

Contact: wangyueran@mail.bnu.edu.cn

Title: Reproducing Trace as State on Local Small Models: Accuracy and Boundary Conditions

Suggested primary category: cs.CL (subject to submitter and moderation review).

Manuscript license: arXiv.org perpetual, non-exclusive license 1.0. This implements the human request for the usual arXiv option; it retains copyright and grants distribution rights to arXiv. It is selected in the submission UI, not applied as a replacement code license. Official basis: https://info.arxiv.org/help/license/index.html

Author block restored from the archived version on human confirmation. No institution-directory identity check is claimed. Account access, any required endorsement, server compilation and final public submission remain human steps.

Abstract (plain text):

Does placing prior reasoning before the original text help a locally served small model answer more accurately? We test Zou and Tang's Trace as State method on official graph questions, retaining the complete source and comparing the same traces before and after it. Two matched twelve-question studies completed 240 calls per model. Qwen3-1.7B prefix F1 was 0.7715 versus 0.7351 for append, with a descriptive placement interval including zero. Qwen3-4B scored 1.0000 in both placements. Its direct F1 was 0.6967, with eighteen of sixty answers output-truncated; a sequential diagnostic with a larger output allowance achieved direct F1 1.0000. Charging all trace preparation, prefix used 5.91 times the tokens of this new direct condition at the same observed accuracy ceiling. A larger context also enabled one previously declined case, while separate trials exposed answer-format sensitivity. The tested configurations therefore did not establish a placement-specific advantage. Output allowance, context fit and scoring rules must be controlled before attributing local performance differences to trace placement. These findings are configuration-specific boundaries, not a universal verdict on the original method.
