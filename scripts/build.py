#!/usr/bin/env python3
"""Generate the README Timeline / Costs tables and the website data from data/timeline.json.

    python3 scripts/build.py          # validate and write the outputs
    python3 scripts/build.py --check  # validate; exit 1 if any output is out of date (CI)

data/timeline.json is the single source of truth. Outputs:
  README.md      the sections between <!-- site|timeline|costs:start --> and <!-- …:end -->
  site/data.js   window.STORYBOARD, read by site/index.html

Standard library only.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "timeline.json"
README = ROOT / "README.md"
SITE_DATA = ROOT / "site" / "data.js"

DATE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def year_of(date):
    p = [int(x) for x in date.split("-")]
    return p[0] + ((p[1] if len(p) > 1 else 6) - 1) / 12 + ((p[2] if len(p) > 2 else 15) - 1) / 365


def validate(d):
    errors, warnings = [], []
    lanes = {l["key"] for l in d["lanes"]}
    seen = set()
    for i, e in enumerate(d["events"]):
        where = f"events[{i}] ({e.get('id', '?')})"
        for k in ("id", "date", "lane", "url", "title", "who", "note"):
            if k not in e:
                errors.append(f"{where}: missing '{k}'")
        if errors:
            continue
        if not SLUG.match(e["id"]):
            errors.append(f"{where}: id must be lowercase-kebab-case")
        if e["id"] in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(e["id"])
        if not DATE.match(e["date"]):
            errors.append(f"{where}: date must be YYYY, YYYY-MM or YYYY-MM-DD")
        for o in e.get("orgs", []):
            if o not in d.get("orgs", {}):
                errors.append(f"{where}: unknown org '{o}' (add it to the top-level 'orgs' registry)")
        if e["lane"] not in lanes:
            errors.append(f"{where}: unknown lane '{e['lane']}' (lanes: {', '.join(sorted(lanes))})")
        if not e["url"].startswith("https://"):
            errors.append(f"{where}: url must be https")
        if e.get("highlight"):
            if not e.get("featured"):
                errors.append(f"{where}: a highlighted event must also be featured")
            if not isinstance(e.get("opened"), dict) or not e["opened"].get("en") or not e["opened"].get("zh"):
                errors.append(f"{where}: a highlighted event needs 'opened' (en and zh): the direction it opened")
        for k in ("title", "who", "note"):
            if not isinstance(e[k], dict) or not e[k].get("en"):
                errors.append(f"{where}: '{k}' needs an 'en' string")
            elif not e[k].get("zh"):
                (errors if e.get("featured") else warnings).append(f"{where}: '{k}' has no 'zh' translation")
    for i, r in enumerate(d.get("recipes", [])):
        for k in ("org", "name", "recipe", "scope", "url"):
            if k not in r:
                errors.append(f"recipes[{i}]: missing '{k}'")
        if r.get("org") not in d.get("orgs", {}):
            errors.append(f"recipes[{i}]: unknown org '{r.get('org')}'")
        if not str(r.get("url", "")).startswith("https://"):
            errors.append(f"recipes[{i}]: url must be https")
    n_hl = sum(1 for e in d["events"] if e.get("highlight"))
    if n_hl > 6:
        warnings.append(f"{n_hl} highlighted events: keep highlights for the few works that opened a whole direction")
    for i, c in enumerate(d["costs"]):
        for k in ("work", "url", "setting", "cost"):
            if k not in c:
                errors.append(f"costs[{i}]: missing '{k}'")
        f = c.get("featured")
        if f:
            for k in ("value", "label", "who", "text"):
                if k not in f:
                    errors.append(f"costs[{i}].featured: missing '{k}'")
    return errors, warnings


def md_escape(s):
    return s.replace("|", "\\|")


def timeline_md(d):
    site = d.get("site_url", "").rstrip("/")
    rows = ["★ Bold rows are the key milestones.", "",
            "| Date | Milestone |", "|---|---|"]
    for e in sorted_events(d):
        date = f"[{e['date']}]({site}/#{e['id']})" if site else e["date"]
        body = f"[{md_escape(e['title']['en'])}]({e['url']}) — {md_escape(e['who']['en'])}: {md_escape(e['note']['en'])}"
        if e.get("highlight"):
            date = f"**{date}**"
            body = (f"**[{md_escape(e['title']['en'])}]({e['url']})** — {md_escape(e['who']['en'])}: "
                    f"{md_escape(e['note']['en'])} ★ *{md_escape(e['opened']['en'])}*")
        rows.append(f"| {date} | {body} |")
    return "\n".join(rows)


def costs_md(d):
    rows = ["| Work | Setting | Stated cost |", "|---|---|---|"]
    for c in d["costs"]:
        rows.append(f"| [{md_escape(c['work'])}]({c['url']}) | {md_escape(c['setting'])} | {md_escape(c['cost'])} |")
    return "\n".join(rows)


def recipes_md(d):
    orgs = d.get("orgs", {})
    rows = ["What each stack turns on to get exact bits, with the cost or limit its own docs state. "
            "Logos identify the organization behind each recipe.", "",
            "| | Who | Turn on | Covers | Stated cost / limit |", "|:-:|---|---|---|---|"]
    for r in d.get("recipes", []):
        o = orgs.get(r["org"], {})
        logo = (f'<img src="https://cdn.simpleicons.org/{o["icon"]}" width="22" height="22" alt="{o["name"]}">' if o.get("icon") else "")
        cost = md_escape(r.get("cost", {}).get("en", ""))
        oname = o.get("name", r["org"])
        who = f"[**{md_escape(oname)}**]({r['url']})" if r["name"] == oname else f"**{md_escape(oname)}** · [{md_escape(r['name'])}]({r['url']})"
        turn_on = f"`{md_escape(r['recipe'])}`" if r.get("code", True) else md_escape(r["recipe"])
        rows.append(f"| {logo} | {who} | {turn_on} | {md_escape(r['scope']['en'])} | {cost} |")
    return "\n".join(rows)


def site_md(d):
    site = d.get("site_url", "").rstrip("/")
    target = f"{site}/" if site else "site/index.html"
    return (f"▶ **[Interactive history]({target})** (English / 简体中文 / bilingual side by side): "
            "a short scroll-through story: why bits differ, the key milestones, and what exactness costs. Every event, with its source, is on the explore page."
            + ("" if site else " It is served by GitHub Pages from `site/`; until Pages is set up, preview it locally (see CONTRIBUTING)."))


def sorted_events(d):
    return [e for _, e in sorted(enumerate(d["events"]), key=lambda p: (year_of(p[1]["date"]), p[0]))]


def site_js(d):
    events = sorted_events(d)
    story = d["story"]
    payload = {
        "title": story["title"], "subtitle": story["subtitle"], "costNote": story["costNote"], "outro": story["outro"],
        "costTrend": story.get("costTrend", []),
        "recipes": d.get("recipes", []),
        "repo": d.get("repo", ""), "site_url": d.get("site_url", ""), "orgs": d.get("orgs", {}),
        "lanes": d["lanes"], "eras": d["eras"],
        "events": events,
        "milestones": [e for e in events if e.get("featured")],
        "costs": d["costs"],
    }
    return ("// GENERATED by scripts/build.py from data/timeline.json. Do not edit; edit the JSON and rebuild.\n"
            "window.STORYBOARD = " + json.dumps(payload, ensure_ascii=False, indent=1) + ";\n")


def replace_section(text, name, body):
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pat.search(text):
        raise SystemExit(f"README.md is missing the {start} … {end} markers")
    note = f"<!-- generated from data/timeline.json by scripts/build.py; edit the JSON, not this table -->"
    return pat.sub(lambda _: f"{start}\n{note}\n\n{body}\n\n{end}", text)


def main():
    check = "--check" in sys.argv[1:]
    d = json.loads(DATA.read_text(encoding="utf-8"))
    errors, warnings = validate(d)
    for w in warnings:
        print(f"warning: {w}")
    if errors:
        for e in errors:
            print(f"error: {e}")
        return 1

    readme_old = README.read_text(encoding="utf-8")
    readme_new = readme_old
    for name, body in (("site", site_md(d)), ("recipes", recipes_md(d)), ("timeline", timeline_md(d)), ("costs", costs_md(d))):
        readme_new = replace_section(readme_new, name, body)
    outputs = {README: (readme_old, readme_new),
               SITE_DATA: (SITE_DATA.read_text(encoding="utf-8") if SITE_DATA.exists() else "", site_js(d))}

    # Every event should also be cited in the curated list, outside the generated tables.
    curated = re.sub(r"<!-- (site|recipes|timeline|costs):start -->.*?<!-- \1:end -->", "", readme_new, flags=re.S)
    for e in d["events"]:
        if e["url"] not in curated:
            print(f"note: '{e['id']}' is in the timeline but its url is not cited in the README list ({e['url']})")

    stale = [p for p, (old, new) in outputs.items() if old != new]
    if check:
        if stale:
            for p in stale:
                print(f"out of date: {p.relative_to(ROOT)} (run: python3 scripts/build.py)")
            return 1
        print(f"ok: {len(d['events'])} events, {sum(1 for e in d['events'] if e.get('featured'))} featured, outputs up to date")
        return 0
    for p in stale:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(outputs[p][1], encoding="utf-8")
        print(f"wrote {p.relative_to(ROOT)}")
    if not stale:
        print("up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
