import re
import unicodedata
import uuid
from planner.models.planner_input import PlannerInput
from planner.models.learning_plan import LearningPlan, PlanItem, TaskAllocation
from planner.core.readiness_evaluator import ReadinessEvaluator
from planner.core.difficulty_selector import DifficultySelector
from planner.core.planning_policy import PlanningPolicy
from planner.core.target_skill_selector import TargetSkillSelector


class PlannerService:
    def __init__(
        self,
        skill_navigator,
        readiness_evaluator: ReadinessEvaluator = None,
        difficulty_selector: DifficultySelector = None
    ):
        self.skill_navigator = skill_navigator
        self.readiness_evaluator = readiness_evaluator or ReadinessEvaluator()
        self.difficulty_selector = difficulty_selector or DifficultySelector()
        self.policy = PlanningPolicy(self.readiness_evaluator, self.difficulty_selector)
        self.target_skill_selector = TargetSkillSelector()

    def create_plan(self, planner_input: PlannerInput) -> LearningPlan:
        if planner_input.selected_domain:
            get_skills = getattr(self.skill_navigator, "get_skills", None)
            available_skills = get_skills() if get_skills else []
            domain_skills = self._skills_in_domain(
                planner_input.selected_domain,
                available_skills,
                planner_input.curriculum_context,
            )
            profile = planner_input.student_profile
            has_domain_history = any(
                skill["id"] in profile.skills
                and profile.skills[skill["id"]].observations > 0
                for skill in domain_skills
            )
            grade_changed = (
                profile.last_grade is not None
                and profile.last_grade != planner_input.curriculum_context.grade
            )
            if domain_skills and (grade_changed or not has_domain_history):
                return self._build_domain_diagnostic_plan(
                    planner_input, domain_skills, grade_changed
                )

        target_id = planner_input.target_skill_id

        if target_id is None:
            get_skills = getattr(self.skill_navigator, "get_skills", None)
            skills = get_skills() if get_skills else []
            assert planner_input.selected_domain is not None
            target_id = self.target_skill_selector.select(
                domain=planner_input.selected_domain,
                skills=skills,
                profile=planner_input.student_profile,
                context=planner_input.curriculum_context,
            )
            if target_id is None:
                # Awaryjny fallback: weź pierwszy skill z działu jeśli istnieje
                domain_fallback = self._skills_in_domain(
                    planner_input.selected_domain,
                    skills,
                    planner_input.curriculum_context,
                )
                if domain_fallback:
                    target_id = domain_fallback[0]["id"]
                else:
                    return LearningPlan(
                        plan_id=str(uuid.uuid4()),
                        target_skill_id="",
                        items=[],
                        status="blocked",
                        reason=(
                            f"Brak nieopanowanych umiejętności w dziale "
                            f"'{planner_input.selected_domain}' dla wybranego kontekstu programu."
                        ),
                    )

        effective_input = planner_input.model_copy(
            update={"target_skill_id": target_id, "selected_domain": None}
        )

        # Krok 1: Walidacja czy cel główny istnieje w grafie (SkillNavigator)
        if not self.skill_navigator.has_skill(target_id):
            return LearningPlan(
                plan_id=str(uuid.uuid4()),
                target_skill_id=target_id,
                items=[],
                status="blocked",
                reason=f"Wybrany skill '{target_id}' nie istnieje w strukturze programu."
            )

        # Krok 2: Pobranie zależności (prerequisites) z SkillNavigatora
        prerequisites = self.skill_navigator.get_prerequisites(target_id)

        # Krok 3: Zbudowanie elementów planu za pomocą PlanningPolicy (wplatanie remediacji/diagnoz)
        items, summary_reason = self.policy.build_items(effective_input, prerequisites)
        for item in items:
            item.subskill_ids = self._select_subskills(item.skill_id, planner_input.student_profile)
            item.subskills_by_skill = {item.skill_id: list(item.subskill_ids)}

        if planner_input.selected_domain:
            get_skills = getattr(self.skill_navigator, "get_skills", None)
            available_skills = get_skills() if get_skills else []
            planned_skill_ids = {item.skill_id for item in items}
            review_skill_id = self.target_skill_selector.select_review_skill(
                planner_input.selected_domain,
                available_skills,
                planner_input.student_profile,
                planner_input.curriculum_context,
                exclude=planned_skill_ids | {target_id},
            )
            secondary_skill_id = review_skill_id or self._select_companion_skill(
                planner_input.selected_domain,
                available_skills,
                planner_input.student_profile,
                planner_input.curriculum_context,
                target_id,
            )
            if secondary_skill_id:
                secondary_role = "review" if review_skill_id else "main"
                secondary_difficulty = self.difficulty_selector.select_difficulty(
                    secondary_skill_id, planner_input.student_profile
                )
                secondary_subskills = self._select_subskills(
                    secondary_skill_id, planner_input.student_profile
                )
                items.append(
                    PlanItem(
                        skill_id=secondary_skill_id,
                        skill_ids=[secondary_skill_id],
                        subskill_ids=secondary_subskills,
                        subskills_by_skill={secondary_skill_id: secondary_subskills},
                        role=secondary_role,
                        target_difficulty=secondary_difficulty,
                        material_scope=(
                            f"{'Powtórka' if review_skill_id else 'Kolejna umiejętność'}: "
                            f"klasa {planner_input.curriculum_context.grade} - "
                            f"{planner_input.curriculum_context.curriculum_level}"
                        ),
                        reason=(
                            "Powtórka opanowanej umiejętności przed zadaniami integrującymi."
                            if review_skill_id
                            else "Druga grupa ćwiczeń przed zadaniami integrującymi."
                        ),
                        task_allocations=[
                            TaskAllocation(task_type="closed_single_choice", count=2)
                        ],
                    )
                )
                integration_skills = [target_id, secondary_skill_id]
                integration_subskills_by_skill = {
                    target_id: self._select_subskills(target_id, planner_input.student_profile),
                    secondary_skill_id: secondary_subskills,
                }
                integration_subskills = list(dict.fromkeys(
                    subskill
                    for values in integration_subskills_by_skill.values()
                    for subskill in values
                ))
                items.append(
                    PlanItem(
                        skill_id=target_id,
                        skill_ids=integration_skills,
                        subskill_ids=integration_subskills,
                        subskills_by_skill=integration_subskills_by_skill,
                        role="integration",
                        target_difficulty=max(
                            self.difficulty_selector.select_difficulty(
                                target_id, planner_input.student_profile
                            ),
                            secondary_difficulty,
                        ),
                        material_scope=(
                            f"Zadanie łączące umiejętności: {', '.join(integration_skills)}; "
                            f"klasa {planner_input.curriculum_context.grade}"
                        ),
                        reason="Po ćwiczeniu osobnych umiejętności plan zawiera zadania integrujące oba cele.",
                        task_allocations=[
                            TaskAllocation(task_type="closed_single_choice", count=2)
                        ],
                    )
                )
                summary_reason += (
                    f" Dodano osobną grupę dla {secondary_skill_id} i zadania łączące "
                    f"ją z umiejętnością {target_id}."
                )

        return LearningPlan(
            plan_id=str(uuid.uuid4()),
            target_skill_id=target_id,
            items=items,
            status="ready",
            reason=summary_reason
        )

    def _build_domain_diagnostic_plan(self, planner_input, skills, grade_changed):
        skill_ids = sorted(skill["id"] for skill in skills)
        items = []
        for skill_id in skill_ids:
            subskills = self._select_subskills(
                skill_id, planner_input.student_profile, limit=1
            )
            items.append(
                PlanItem(
                    skill_id=skill_id,
                    subskill_ids=subskills,
                    subskills_by_skill={skill_id: subskills},
                    role="diagnostic",
                    target_difficulty=self.difficulty_selector.select_difficulty(
                        skill_id, planner_input.student_profile
                    ),
                    material_scope=(
                        f"Diagnoza umiejętności {skill_id}: klasa "
                        f"{planner_input.curriculum_context.grade}"
                    ),
                    reason=(
                        "Jedno zadanie diagnostyczne po zmianie klasy."
                        if grade_changed
                        else "Jedno zadanie diagnostyczne, ponieważ uczeń nie ma jeszcze "
                        "wyników w tym dziale."
                    ),
                    task_allocations=[
                        TaskAllocation(task_type="closed_single_choice", count=1)
                    ],
                )
            )

        trigger_reason = (
            "zmianę klasy"
            if grade_changed
            else "brak wcześniejszych wyników w dziale"
        )
        return LearningPlan(
            plan_id=str(uuid.uuid4()),
            target_skill_id=skill_ids[0],
            items=items,
            status="needs_diagnostic",
            reason=(
                f"Przygotowano diagnozę działu '{planner_input.selected_domain}': "
                f"po jednym zadaniu dla każdej z {len(items)} umiejętności "
                f"z powodu {trigger_reason}."
            ),
        )

    def _skills_in_domain(self, domain, skills, context):
        domain_key = self._normalize_domain(domain)
        matched = [
            skill
            for skill in skills
            if self.target_skill_selector._matches_context(skill, context)
            and any(
                self._normalize_domain(tag) == domain_key
                for tag in skill.get("tags", [])
            )
        ]
        if matched:
            return matched

        # Fallback: dopasuj po samym dziale jeśli filtr kontekstu nie zwrócił żadnego skilla
        return [
            skill
            for skill in skills
            if any(
                self._normalize_domain(tag) == domain_key
                for tag in skill.get("tags", [])
            )
        ]

    def _select_subskills(self, skill_id, profile, limit=2):
        definitions = getattr(self.skill_navigator, "skills", {})
        if isinstance(definitions, dict):
            skill = definitions.get(skill_id, {})
        else:
            get_skills = getattr(self.skill_navigator, "get_skills", None)
            skill = next(
                (value for value in get_skills() if value.get("id") == skill_id),
                {},
            ) if get_skills else {}
        objectives = skill.get("objectives", [])
        state = profile.skills.get(skill_id)
        subskill_states = getattr(state, "subskills", {}) if state else {}
        ordered = sorted(
            objectives,
            key=lambda objective: (
                subskill_states.get(objective["id"]).observations
                if subskill_states.get(objective["id"]) else 0,
                subskill_states.get(objective["id"]).ability
                if subskill_states.get(objective["id"]) else float("-inf"),
            ),
        )
        return [objective["id"] for objective in ordered[:limit]]

    def _select_companion_skill(self, domain, skills, profile, context, target_id):
        domain_key = self._normalize_domain(domain)
        candidates = []
        for skill in skills:
            skill_id = skill["id"]
            if skill_id == target_id or not self.target_skill_selector._matches_context(skill, context):
                continue
            if not any(
                self._normalize_domain(tag) == domain_key
                for tag in skill.get("tags", [])
            ):
                continue
            prerequisites = skill.get("prerequisites", [])
            if not all(
                prerequisite == target_id
                or self.readiness_evaluator.evaluate(prerequisite, profile) == "mastered"
                for prerequisite in prerequisites
            ):
                continue
            state = profile.skills.get(skill_id)
            mastered = bool(
                state
                and state.observations >= self.target_skill_selector.minimum_observations_for_mastery
                and self.readiness_evaluator.evaluate(skill_id, profile) == "mastered"
            )
            candidates.append((not mastered, skill_id))
        candidates.sort()
        return candidates[0][1] if candidates else None

    @staticmethod
    def _normalize_domain(value):
        if not value:
            return ""
        decomposed = unicodedata.normalize("NFKD", str(value).casefold())
        without_diacritics = "".join(
            char for char in decomposed if not unicodedata.combining(char)
        ).replace("\u0142", "l")
        # Usuń rozszerzenie .json oraz wiodący prefiks typu "01-" / "02_"
        cleaned = re.sub(r"\.json$", "", without_diacritics.strip())
        cleaned = re.sub(r"^\d+[-_]+", "", cleaned)

        aliases = {
            "ulamki": "fractions",
            "ulamkow": "fractions",
            "fraction": "fractions",
            "fractions": "fractions",
            "arytmetyka": "arithmetic",
            "arithmetic": "arithmetic",
        }
        return aliases.get(cleaned, cleaned)
