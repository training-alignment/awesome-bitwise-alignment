# Website

Deployed to GitHub Pages by `.github/workflows/pages.yml`. Two pages share the generated `data.js` (built by `scripts/build.py` from `data/timeline.json`; do not edit it):

- **`index.html`, the story.** Ten full-screen slides you scroll through like a keynote. Each new slide rises over the previous one, which shrinks, blurs and dims beneath it.
  - The slides: the cover; the problem (two summation trees for `(0.1+0.2)+0.3` vs `0.1+(0.2+0.3)`, computed live, with the hex results scrambling into place); how it evolved (every event on one line, with the key milestones lit and linked to cards); one key frame per `highlight` event; the bitwise recipes that ship today (`recipes`); the falling cost of exactness (`story.costTrend`); and a closing slide.
  - Key frames show the organization's logo (Simple Icons, pinned CDN version) or a monogram, a large outline year, and the direction the work opened.
  - Navigation: the keyboard (↑/↓, Space, PgUp/PgDn, Home/End), the side dots, or deep links `#<slide-id>`, e.g. `#palm-bitwise` or `#s-cost`.
- **`explore.html`, everything.** All events with a pinned swimlane chart that follows the card you are reading. Filter by lane or by ★ key milestones; deep links are `#<event-id>`.

Both pages:
- English by default; 简体中文 or bilingual side by side via the switch or `?lang=zh|both` (a choice made with the switch is remembered).
- Light and dark themes, a reduced-motion fallback, and a phone layout.
- Fonts from Google Fonts (Inter, Noto Sans SC, JetBrains Mono), with system fallbacks.

Open them over http (`python3 -m http.server --directory site`), not as `file://`: some viewers render local files without JavaScript, and both pages are drawn by JavaScript.

Logos are trademarks of their owners and are used only to identify the organization behind each milestone.
