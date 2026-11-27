import re
import unicodedata

from planner.models.planner_input import CurriculumContext
from profiler.core.irt_model import calculate_difficulty_mastery
from profiler.models.student_profile import StudentProfile


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    without_diacritics = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    ).replace("\u0142", "l")
    cleaned = re.sub(r"\.json$", "", without_diacritics.strip())
    return re.sub(r"^\d+[-_]+", "", cleaned)


def _canonical_stage(value: str) -> str:
    normalized = _normalize(value)
    if any(
        token in normalized
        for token in ("secondary", "liceum", "highschool", "ponadpodstaw")
    ):
        return "secondary"
    if any(token in normalized for token in ("primary", "podstaw", "elementary")):
        return "primary"
    return normalized


def _canonical_level(value: str) -> str:
    normalized = _normalize(value)
    if normalized in {"podstawa", "standard", "basic"}:
        return "standard"
    if normalized in {"rozszerzenie", "extension", "advanced"}:
        return "extension"
    return normalized


def _canonical_domain(value: str) -> str:
    normalized = _normalize(value)
    aliases = {
        "ulamki": "fractions",
        "ulamkow": "fractions",
        "fraction": "fractions",
        "fractions": "fractions",
        "arytmetyka": "arithmetic",
        "arithmetic": "arithmetic",
    }
    return aliases.get(normalized, normalized)


class TargetSkillSelector:
    """Select the next unmastered skill in a learner-selected curriculum domain."""

    minimum_observations_for_mastery = 3
    review_mastery_threshold = 0.7
    minimum_mastered_skills_for_interleaving = 1

    def select(
        self,
        domain: str,
        skills: list[dict],
        profile: StudentProfile,
        context: CurriculumContext,
    ) -> str | None:
        domain_key = _canonical_domain(domain)
        candidates = [
            skill
            for skill in skills
            if any(
                _canonical_domain(tag) == domain_key
                for tag in skill.get("tags", [])
            )
            and self._matches_context(skill, context)
        ]
        if not candidates:
            candidates = [
                skill
                for skill in skills
                if any(
                    _canonical_domain(tag) == domain_key
                    for tag in skill.get("tags", [])
                )
            ]
        if not candidates:
            return None

        by_id = {skill["id"]: skill for skill in skills}
        depths = {}

        def dependency_depth(skill_id: str, visiting: set[str] | None = None) -> int:
            if skill_id in depths:
                return depths[skill_id]
            visiting = set() if visiting is None else visiting
            if skill_id in visiting:
                return 0
            visiting.add(skill_id)
            prerequisites = by_id.get(skill_id, {}).get("prerequisites", [])
            depth = (
                1
                + max(
                    (dependency_depth(item, visiting.copy()) for item in prerequisites if item in by_id),
                    default=0,
                )
                if prerequisites
                else 0
            )
            depths[skill_id] = depth
            return depth

        ordered = sorted(
            candidates,
            key=lambda skill: (dependency_depth(skill["id"]), skill["id"]),
        )
        for skill in ordered:
            state = profile.skills.get(skill["id"])
            if (
                state is None
                or state.observations < self.minimum_observations_for_mastery
                or state.uncertainty > 1.5
            ):
                return skill["id"]

            working_level = state.working_difficulty or 1
            mastery = calculate_difficulty_mastery(
                state.ability,
                state.uncertainty,
            )[working_level]
            if mastery < 0.7:
                return skill["id"]

        return ordered[0]["id"]

    def select_review_skill(
        self,
        domain: str,
        skills: list[dict],
        profile: StudentProfile,
        context: CurriculumContext,
        *,
        exclude: set[str] | None = None,
    ) -> str | None:
        """Choose the least-recently practiced mastered skill for spaced review."""
        domain_key = _canonical_domain(domain)
        excluded = exclude or set()
        mastered = []
        for skill in skills:
            skill_id = skill["id"]
            if skill_id in excluded or not self._matches_context(skill, context):
                continue
            if not any(
                _canonical_domain(tag) == domain_key
                for tag in skill.get("tags", [])
            ):
                continue
            state = profile.skills.get(skill_id)
            if (
                state is None
                or state.observations < self.minimum_observations_for_mastery
                or state.uncertainty > 1.5
            ):
                continue
            level = state.working_difficulty or 1
            mastery = calculate_difficulty_mastery(
                state.ability, state.uncertainty
            )[level]
            if mastery >= self.review_mastery_threshold:
                mastered.append((state.last_practiced_at, skill_id))
        if len(mastered) < self.minimum_mastered_skills_for_interleaving:
            return None
        mastered.sort(key=lambda item: (item[0], item[1]))
        return mastered[0][1]

    @staticmethod
    def _matches_context(skill: dict, context: CurriculumContext) -> bool:
        curriculum = skill.get("curriculum", {})
        introduced_grade = curriculum.get("introduced_in_grade")
        if introduced_grade is not None and introduced_grade > context.grade:
            return False

        school_type = curriculum.get("school_type")
        if school_type and _canonical_stage(school_type) != _canonical_stage(
            context.education_stage
        ):
            return False

        level = curriculum.get("level")
        return not level or _canonical_level(level) == _canonical_level(
            context.curriculum_level
        )
