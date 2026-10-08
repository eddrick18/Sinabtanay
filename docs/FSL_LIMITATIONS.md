# Scope, accessibility, and privacy

The intended project is a **Real-Time Filipino Sign Language Gesture Recognition
System for a Selected Vocabulary**. Phase 2 displays webcam frames, detects hands,
and draws landmarks. It has no sign classification, trained FSL model, or sign
recognition accuracy. Its pretrained MediaPipe model estimates hand locations.
Phase 3 converts these locations into 126 wrist-relative, scale-normalized
features. Independent hand normalization discards absolute location and relative
wrist separation, limiting suitability for signs distinguished by those cues.

FSL is a complete natural language involving hand shape, orientation, movement,
location, two-hand interaction, facial expressions, body movement, timing,
grammar, and context. Isolated static hand features cannot represent all of it.
Future labels such as HELLO or THANK_YOU are placeholders, not verified gestures.
Actual training signs must be verified with reliable FSL references or qualified
FSL users, interpreters, teachers, or Deaf collaborators. Never invent gestures.

This project does not replace interpreters, FSL education, or Deaf communication
professionals. Future recognition must expose uncertainty and use UNKNOWN when
appropriate; confidence estimates must not be presented as guaranteed accuracy.

Phases 1–3 process frames locally in memory. They do not record images or video
or upload webcam footage. Later dataset collection should prefer landmarks and
require informed consent from participants; numerical landmarks still deserve
careful handling.
