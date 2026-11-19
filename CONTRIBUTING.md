# Contributing

Thanks for helping. This list is only useful if every entry is real, primary, and checkable.

## What belongs here

Work on **bitwise reproducibility, determinism, batch invariance, and train–inference alignment** of ML (mostly LLM) systems:
papers, blog posts, official docs, framework flags, GitHub repos, and issues/PRs that carry a design decision or a measurement.
Generic "set your seed" tutorials, marketing pages, and reposts of a primary source do not.

## Entry format

One line per entry, in the section that fits best:

```markdown
- [Title](https://primary-source-url) — Authors or Org · YYYY-MM-DD · What it does, in one line. Cost/limits if the source states them.
```

- **Link the primary source**: the arXiv abstract page, the conference page, the official blog or docs, or the GitHub repo/issue. Not a tweet about it.
- **Date**: publication date (`YYYY-MM-DD` or `YYYY-MM`). Use `living doc` for documentation pages without a date.
- **Numbers only from the source.** If you quote an overhead ("~25% per step"), it must appear in the linked page, with its setup. Never estimate one.
- **No adjectives** ("amazing", "state-of-the-art"). Say what it does and what it costs.
- Keep entries within a section in **chronological order** (oldest first) unless the section says otherwise.
- If a claim is your reading rather than the source's words, say so ("our reading: …").

## Timeline, costs and the website

[`data/timeline.json`](data/timeline.json) is the single source of truth for three things: the README **Timeline** table, the README **What exactness costs** table, and the website in [`site/`](site/). Do not edit those tables or `site/data.js` by hand. Edit the JSON, then regenerate:

```bash
python3 scripts/build.py
```

The script validates the data, rewrites the README sections between the `<!-- …:start -->` / `<!-- …:end -->` markers, and writes `site/data.js`. CI runs `python3 scripts/build.py --check` and fails if the generated files are out of date.

Each event needs `id` (kebab-case, also its deep link `#id` on the website), `date`, `lane`, `url` and bilingual `title` / `who` / `note` (`en` required; `zh` strongly encouraged). Optional fields:

- `featured: true` gives the event its own card on the explore page (`site/explore.html`). Featured events need `zh` text.
- `orgs: [...]` lists keys from the top-level `orgs` registry; the website shows their logos ([Simple Icons](https://simpleicons.org) slugs) or a monogram. Only list an organization the primary source names.
- `highlight: true` with `opened: {en, zh}` makes the event a **key milestone**: a full-screen key frame in the story deck (`site/index.html`) and a bold ★ row in the README. Key milestones are deliberately rare, one per direction the field opened, so a new one needs a strong case in the PR. Highlighted events must also be featured.

Preview the website locally:

```bash
python3 -m http.server 8000 --directory site
```

Then open `http://localhost:8000/`. Add `?lang=zh` or `?lang=both`, or `#<event-id>` to jump to an event.

When a push to `main` lands, [`.github/workflows/pages.yml`](.github/workflows/pages.yml) deploys `site/` to GitHub Pages. Once Pages is live, set `site_url` (and optionally `repo`) in the JSON and rebuild: README dates then link to the matching moment on the website.

## Checking links

```bash
python3 scripts/check_links.py README.md
```

Please run it before opening a PR. Some sites (e.g. OpenAI, some publishers) block scripted requests; the script reports those as `blocked` rather than `dead`, so check them in a browser.

## Pull requests

- One topic per PR; say in the description how you verified each entry.
- Corrections (wrong date, wrong author, dead link, an overhead that the source does not state) are as welcome as new entries.

By contributing you agree to dedicate your contribution to the public domain under [CC0 1.0](LICENSE).
