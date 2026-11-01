"""Minimal inference subset of VITS (https://github.com/jaywalnut310/vits, MIT License).

Vendored unmodified except that imports are package-relative, ``text.cleaners`` (which needs
phonemizer) is imported only when text is cleaned, and ``monotonic_align``
(a Cython extension used only when training VITS) is imported lazily.
Brain2Speech-Net uses a frozen, pretrained LJSpeech VITS: its text encoder supplies the
contextual phoneme representations (CPRs) and its duration predictor, flow and decoder
turn predicted CPRs into a waveform.
"""
