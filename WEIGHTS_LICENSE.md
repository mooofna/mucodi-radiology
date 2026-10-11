# Licence of the MuCoDi student weights

This file governs the trained weights of the fourteen MuCoDi students (TMIN, T0 to T6, B0 and B2
to B6) in every form this project publishes them: the ONNX files served by the web demo and
attached to the GitHub release, and any PyTorch checkpoint shared with them. The source code is
licensed separately (`LICENSE`, CC BY-NC 4.0). The scans bundled with the demo are covered by
`DATA_LICENSE.md`.

Copyright (c) 2026 Maurice Heide.

## Why these terms

The students were trained to reproduce the embeddings of two frozen teachers:

- **Curia-1** (Raidium, <https://huggingface.co/raidium/curia>), released under the Curia
  Research-Only RAIL-M licence, reproduced in full in
  [`licenses/CURIA_RESEARCH_RAIL-M.md`](licenses/CURIA_RESEARCH_RAIL-M.md). The students are
  treated as Derivatives of Curia-1 in the sense of that licence, so its permitted purpose
  (academic or research use only) and its use-based restrictions carry over to them.
- **Pillar-0** (Yala Lab, <https://huggingface.co/YalaLab/Pillar0-ChestCT>), released under the
  Educational Community License 2.0 ([`licenses/ECL-2.0.txt`](licenses/ECL-2.0.txt)). Its
  embeddings served as distillation targets; no Pillar-0 weights are included.

## Terms

1. **Permitted purpose.** You may use, copy, modify and redistribute the weights for academic or
   research purposes only. Commercial use is not permitted.
2. **Use restrictions.** You may not use the weights, or any model derived from them, in any of
   the ways listed in Attachment A below, which is Attachment A of the Curia Research-Only RAIL-M
   licence, reproduced verbatim.
3. **Passing the terms on.** If you share the weights or a model derived from them, including as
   a hosted service, you must include these terms (this file) and
   `licenses/CURIA_RESEARCH_RAIL-M.md` with them, make the use restrictions a condition of
   whatever licence you share them under, and mark any files you changed.
4. **Not for clinical use.** The weights are not a medical device and have not been cleared or
   approved for any clinical purpose. They must not be used to provide medical advice or to make
   clinical decisions.
5. **Attribution.** Cite the thesis (see `README.md`) and the publications of both teachers.
6. **No warranty.** The weights are provided as is, without warranty of any kind, express or
   implied. In no event shall the authors be liable for any claim or damages arising from their
   use.

## Attachment A: use restrictions (verbatim from the Curia Research-Only RAIL-M licence)

In the text below, "the Artifact or its Derivatives" includes the MuCoDi student weights.

You agree not to use the Artifact or its Derivatives in any of the following ways:

1. Discrimination

   (a) To discriminate or exploit individuals or groups based on legally protected characteristics and/or vulnerabilities.

   (b) For purposes of administration of justice, law enforcement, immigration, or asylum processes, such as predicting that a natural person will commit a crime or the likelihood thereof.

2. Military

   (a) For weaponry or warfare.

3. Legal

   (a) To engage or enable fully automated decision-making that adversely impacts a natural person's legal rights without expressly and intelligibly disclosing the impact to such natural person and providing an appeal process.

   (b) To engage or enable fully automated decision-making that creates, modifies or terminates a binding, enforceable obligation between entities; whether these include natural persons or not.

4. Disinformation

   (a) To create, present or disseminate verifiably false or misleading information for economic gain or to intentionally deceive the public, including creating false impersonations of natural persons.

   (b) To synthesize or modify a natural person's appearance, voice, or other individual characteristics, unless prior informed consent of said natural person is obtained.

   (c) To autonomously interact with a natural person, in text or audio format, unless disclosure and consent is given prior to interaction that the system engaging in the interaction is not a natural person.

5. Privacy

   (a) To utilize personal information to infer additional personal information about a natural person, including but not limited to legally protected characteristics, vulnerabilities or categories; unless informed consent from the data subject to collect said inferred personal information for a stated purpose and defined duration is received.

6. Health

   (a) To provide medical advice or make clinical decisions without necessary (external) accreditation of the system; unless the use is (i) in an internal research context with independent and accountable oversight and/or (ii) with medical professional oversight that is accompanied by any related compulsory certification and/or safety/quality standard for the implementation of the technology.
