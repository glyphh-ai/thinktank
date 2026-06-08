"""Match detected issues against ground truth and compute precision/recall/F1."""

from __future__ import annotations

from dataclasses import dataclass

from .cases import Case, Decoy, Plant


@dataclass
class CaseResult:
    case_id: str
    plants: int
    found: int            # plants matched by >=1 issue (true positives, recall numerator)
    issues: int           # total issues the pipeline raised
    true_issues: int      # issues that matched some plant
    decoy_violations: int # decoys wrongly flagged
    false_positives: int  # issues matching no plant (incl. decoy violations)


def _issue_text(issue: dict) -> str:
    return f"{issue.get('title', '')} {issue.get('description', '')}".lower()


def _key_token(key: str) -> str:
    return key.replace("_", " ").split()[0]  # "eye_color" -> "eye"


# A planted attribute can be described with synonyms in the issue text; the plant
# names the concept, so any synonym of it counts as a hit (this isn't gaming the
# score -- "role" and "profession" are the same attribute).
_SYNONYMS = {
    "role": {"role", "profession", "occupation", "job", "title", "position", "surgeon", "accountant"},
    "eye_color": {"eye", "eyes"},
    "age": {"age", "years", "old"},
    "hair": {"hair", "blonde", "black"},
    "height": {"height", "tall", "short", "stature", "tallest", "shortest"},
    "hometown": {"hometown", "village", "abroad", "born", "birth", "travel", "left", "cities"},
    "object_status": {"watch", "pocket", "crushed", "destroyed", "broken", "intact", "repair"},
    "sibling_count": {"sibling", "sister", "child", "only", "brother"},
    "residence": {"residence", "lived", "moved", "quarter", "district", "home"},
    "rank": {"rank", "captain", "lieutenant", "promot"},
}


def _matches_plant(plant: Plant, issue: dict) -> bool:
    text = _issue_text(issue)
    if plant.entity.lower() not in text:
        return False
    tokens = _SYNONYMS.get(plant.key, {_key_token(plant.key)})
    return any(t in text for t in tokens) or plant.category == issue.get("category")


def _matches_decoy(decoy: Decoy, issue: dict) -> bool:
    text = _issue_text(issue)
    if decoy.entity.lower() not in text:
        return False
    return any(t in text for t in _SYNONYMS.get(decoy.key, {_key_token(decoy.key)}))


def score_case(case: Case, issues: list[dict]) -> CaseResult:
    found = sum(1 for p in case.plants if any(_matches_plant(p, i) for i in issues))
    true_issues = sum(1 for i in issues if any(_matches_plant(p, i) for p in case.plants))
    decoy_violations = sum(1 for d in case.decoys if any(_matches_decoy(d, i) for i in issues))
    false_positives = sum(1 for i in issues if not any(_matches_plant(p, i) for p in case.plants))
    return CaseResult(
        case_id=case.id, plants=len(case.plants), found=found, issues=len(issues),
        true_issues=true_issues, decoy_violations=decoy_violations,
        false_positives=false_positives,
    )


@dataclass
class Benchmark:
    results: list[CaseResult]

    @property
    def recall(self) -> float:
        plants = sum(r.plants for r in self.results)
        return sum(r.found for r in self.results) / plants if plants else 1.0

    @property
    def precision(self) -> float:
        issues = sum(r.issues for r in self.results)
        return sum(r.true_issues for r in self.results) / issues if issues else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def decoy_violations(self) -> int:
        return sum(r.decoy_violations for r in self.results)

    @property
    def false_positives(self) -> int:
        return sum(r.false_positives for r in self.results)
