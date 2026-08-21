# 📋 VinylVision Roadmap & TODO

This document tracks planned features, enhancements, and technical debt for upcoming releases of VinylVision.

---

## 🚀 Future Features & Integrations

- [ ] **Local LLM / Vinyl Trivia Assistant**
  - Add an optional background worker that fetches interesting trivia, recording history, and fun facts about the recognized vinyl release.

## 🖥️ UI / UX & Kiosk Enhancements

- [ ] **Dedicated Kiosk Mode**
  - Add `--kiosk` / `--fullscreen` CLI flag to hide window decorations and launch borderless on dedicated displays (e.g., Raspberry Pi Touch Displays).
- [ ] **Standby & Screensaver Mode**
  - Displays a full-screen ambient slideshow cycling through random album covers and artist info from the local collection database when idle (no vinyl on stand for $N$ minutes).
  - Wakes up instantly on touch/click.
- [ ] **Smooth Lyrics Scrolling**
  - Implement smooth transition animations between active lyric lines instead of instant text jumps.
- [ ] **Touchscreen Gesture Controls**
  - Add tap-to-refresh, swipe-to-switch views, and touch-friendly settings controls.

## 👁️ Computer Vision & Dataset

- [x] **Automatic Perspective Calibration**
  - Automatic detection of stand corners using contour/edge detection or a lightweight keypoint model to eliminate manual corner picking.
- [ ] **Adaptive Lighting Normalization**
  - Implement CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing to improve cover matching under harsh ambient reflections or low light.
