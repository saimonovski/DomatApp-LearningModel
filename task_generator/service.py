from __future__ import annotations

import json
import os
import random
import time
import unicodedata
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from planner.models.learning_plan import TaskAllocation
from planner.models.planner_input import CurriculumContext
from task_generator.models import (
    AnswerOption,
    PresentedOption,
    PresentedTask,
    TaskGenerationRequest,
    TaskTemplate,
)
from task_generator.quality import has_single_true_comparison, is_statement_well_formed

GENERATOR_VERSION = "0.2.0"
TASK_PROMPT_VERSION = "closed-task-stream-v2"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
RETRYABLE_GEMINI_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _urlopen_with_retries(request, *, timeout, max_attempts=3):
    """Retry transient Gemini capacity and transport errors with backoff."""
    for attempt in range(max_attempts):
        try:
            return urlopen(request, timeout=timeout)
        except HTTPError as error:
            if error.code not in RETRYABLE_GEMINI_STATUS_CODES or attempt + 1 == max_attempts:
                raise
            retry_after = error.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2**attempt
            except ValueError:
                delay = 2**attempt
            error.close()
            time.sleep(min(delay, 8))
        except URLError:
            if attempt + 1 == max_attempts:
                raise
            time.sleep(2**attempt)


def _extract_json_array_items(buffer, cursor):
    """Return complete objects decoded so far from a streamed JSON array."""
    decoder = json.JSONDecoder()
    if cursor == 0:
        start = buffer.find("[")
        if start < 0:
            return [], cursor, False
        cursor = start + 1

    items = []
    while cursor < len(buffer):
        while cursor < len(buffer) and (buffer[cursor].isspace() or buffer[cursor] == ","):
            cursor += 1
        if cursor >= len(buffer):
            break
        if buffer[cursor] == "]":
            return items, cursor + 1, True
        try:
            item, end = decoder.raw_decode(buffer, cursor)
        except json.JSONDecodeError:
            break
        if not isinstance(item, dict):
            raise ValueError("Each streamed task must be a JSON object")
        items.append(item)
        cursor = end
    return items, cursor, False


class TaskGenerationError(ValueError):
    """A request cannot be fulfilled without violating its plan constraints."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class PresentationKeyStore(Protocol):
    def save(
        self,
        presentation_id: str,
        correct_option_id: str,
        option_source_ids: dict[str, str],
        *,
        task_id: str | None = None,
        presented_task: dict | None = None,
    ) -> None: ...

    def get(self, presentation_id: str) -> dict | None: ...


class InMemoryPresentationKeyStore:
    """Development/test store; production should provide a persistent store."""

    def __init__(self):
        self._keys = {}

    def save(
        self,
        presentation_id,
        correct_option_id,
        option_source_ids,
        *,
        task_id=None,
        presented_task=None,
    ):
        self._keys[presentation_id] = {
            "correct_option_id": correct_option_id,
            "option_source_ids": dict(option_source_ids),
            "task_id": task_id,
            "presented_task": presented_task,
        }

    def get(self, presentation_id):
        return self._keys.get(presentation_id)


class TaskGeneratorService:
    """Generate deterministic, machine-checkable single-choice math templates."""

    def __init__(
        self,
        skill_navigator,
        rng=None,
        *,
        gemini_api_key: str | None = None,
        gemini_model: str | None = None,
    ):
        self.skill_navigator = skill_navigator
        self.rng = rng or random.SystemRandom()
        env_key = os.getenv("GEMINI_API_KEY")
        self.gemini_api_key = (gemini_api_key or env_key or "").strip() or None
        raw_model = gemini_model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.gemini_model = raw_model.removeprefix("models/")
        get_skills = getattr(skill_navigator, "get_skills", None)
        self.skills = {
            skill["id"]: skill for skill in (get_skills() if get_skills else [])
        }

    def generate(
        self,
        request: TaskGenerationRequest,
        *,
        seed: int | None = None,
    ) -> TaskTemplate:
        item = request.plan_item
        skill_id = item.skill_id
        skill_ids = item.skill_ids or [skill_id]
        if any(not self.skill_navigator.has_skill(value) for value in skill_ids):
            raise TaskGenerationError(
                "unknown_skill", f"Skill '{skill_id}' is not in the skill tree."
            )
        if any(value not in self.skills for value in skill_ids):
            raise TaskGenerationError(
                "missing_skill_definition",
                f"No curriculum definition is available for '{skill_id}'.",
            )
        for value in skill_ids:
            self._validate_context(value, request.curriculum_context)
        self._validate_plan_subskills(item)

        if seed is not None and seed < 0:
            raise TaskGenerationError(
                "invalid_seed", "Generation seed must be non-negative."
            )
        generation_seed = seed if seed is not None else self.rng.randrange(0, 2**63)
        difficulty = int(round(item.target_difficulty))
        if not self.gemini_api_key:
            raise TaskGenerationError(
                "missing_gemini_api_key",
                "Set GEMINI_API_KEY before generating tasks.",
            )
        statement, correct, distractors, steps, explanation = self._generate_with_gemini(
            request, generation_seed
        )

        answers = self._build_answers(correct, distractors)
        correct_ids = [
            answer.id for answer in answers if answer.text in set(correct)
        ]
        return TaskTemplate(
            task_id=str(uuid4()),
            plan_id=request.plan_id,
            skill_id=skill_id,
            skill_ids=skill_ids,
            subskill_ids=item.subskill_ids,
            subskills_by_skill=item.subskills_by_skill,
            difficulty=difficulty,
            requested_difficulty=item.target_difficulty,
            curriculum_context=request.curriculum_context,
            material_scope=item.material_scope,
            statement=statement,
            answers=answers,
            correct_answers=correct_ids,
            solution_steps=steps,
            explanation=explanation,
            generation_seed=generation_seed,
            validation_checks=[
                "skill_exists",
                "curriculum_context",
                "answer_pool_unique",
                "correct_answer_in_pool",
                "distractor_count",
                "gemini_self_check_prompt",
            ],
            generator_version=GENERATOR_VERSION,
        )

    def generate_batch(self, request: TaskGenerationRequest, *, seed: int | None = None) -> list[TaskTemplate]:
        """Collect every planned task from the streaming Gemini response."""
        return list(self.generate_batch_stream(request, seed=seed))

    def generate_batch_stream(
        self,
        request: TaskGenerationRequest,
        *,
        seed: int | None = None,
        _partial_attempt: int = 0,
    ):
        """Yield each complete task as soon as it arrives from Gemini."""
        skill_id = request.plan_item.skill_id
        skill_ids = request.plan_item.skill_ids or [skill_id]
        if any(not self.skill_navigator.has_skill(value) for value in skill_ids):
            raise TaskGenerationError("unknown_skill", f"Skill '{skill_id}' is not in the skill tree.")
        if any(value not in self.skills for value in skill_ids):
            raise TaskGenerationError("missing_skill_definition", f"No curriculum definition is available for '{skill_id}'.")
        for value in skill_ids:
            self._validate_context(value, request.curriculum_context)
        self._validate_plan_subskills(request.plan_item)
        if not self.gemini_api_key:
            raise TaskGenerationError("missing_gemini_api_key", "Set GEMINI_API_KEY before generating tasks.")
        if seed is not None and seed < 0:
            raise TaskGenerationError("invalid_seed", "Generation seed must be non-negative.")

        allocations = request.plan_item.task_allocations
        if any(allocation.task_type != "closed_single_choice" for allocation in allocations):
            unsupported = next(
                allocation.task_type for allocation in allocations
                if allocation.task_type != "closed_single_choice"
            )
            raise TaskGenerationError("unsupported_task_type", f"Task type '{unsupported}' is not supported yet.")
        task_count = sum(allocation.count for allocation in allocations)
        if not 1 <= task_count <= 10:
            raise TaskGenerationError("invalid_batch_size", "A Gemini batch must contain between 1 and 10 tasks.")

        batch_seed = seed if seed is not None else self.rng.randrange(0, 2**63)
        task_schema = {
            "type": "OBJECT",
            "properties": {
                "statement": {
                    "type": "STRING",
                    "description": (
                        "One short, direct Polish instruction, preferably 5-15 words. "
                        "No story, real-life setup, or multiple questions."
                    ),
                },
                "correct_answer": {"type": "STRING"},
                "distractors": {"type": "ARRAY", "items": {"type": "STRING"}},
                "solution_steps": {"type": "ARRAY", "items": {"type": "STRING"}},
                "explanation": {"type": "STRING"},
            },
            "required": ["statement", "correct_answer", "distractors", "solution_steps", "explanation"],
        }
        prompt = {
            "instruction": (
                "Wygeneruj żądaną liczbę różnych, jednoznacznych zadań matematycznych po polsku. "
                "To są krótkie zadania zamknięte. Docelową klasę odczytaj z zakresu materiału "
                "w planie; nie obniżaj poziomu tylko dlatego, że skill wprowadzono wcześniej. "
                "Treść każdego "
                "zadania ma być jednym bezpośrednim poleceniem, najlepiej 5-15 słów. Nie pisz "
                "historyjek, opisów sytuacji życiowych, wstępów ani kilku pytań naraz. Nie dodawaj "
                "zbędnych danych ani prośby o uzasadnienie w treści zadania. Trudność buduj przez "
                "dobór liczb i operacji odpowiednich do celu oraz poziomu z planu, a nie przez "
                "długi opis. Kieruj się celami i zakresem skillu. Zwróć wyłącznie gotową treść zadania; "
                "nie pokazuj wersji roboczych, autokorekty ani tekstów typu 'Nie, wskaż...'. "
                "Treść ma zawierać tylko jedno polecenie. Przed zwróceniem każdego zadania "
                "rozwiąż je samodzielnie i sprawdź poprawność odpowiedzi oraz błędność dystraktorów. "
                "Podaj dokładnie trzy różne wiarygodne błędne odpowiedzi i kroki rozwiązania. "
                "W zadaniach porównawczych sprawdź prawdziwość każdego wariantu; nie podawaj relacji "
                "równoważnych poprawnej odpowiedzi, np. >= jako dystraktora dla >. "
                "nie powtarzaj treści ani zestawów liczb w paczce. Przykładowa forma: "
                "Zamień podany ułamek na liczbę mieszaną. Dobierz konkretne dane do skillu."
            ),
            "skill_guidelines": [self.skills[value] for value in skill_ids],
            "primary_skill_id": skill_id,
            "integrated_skills": skill_ids,
            "target_subskills": request.plan_item.subskill_ids,
            "subskills_by_skill": request.plan_item.subskills_by_skill,
            "integration_instruction": "If multiple skills are listed, create a task that requires their combined use. Target only the listed subskills for each skill.",
            "plan": request.plan_item.model_dump(mode="json"),
            "curriculum_context": request.curriculum_context.model_dump(mode="json"),
            "difficulty": round(request.plan_item.target_difficulty),
            "task_count": task_count,
            "task_variation_instruction": "Within this batch vary the cognitive form across tasks (for example symbolic notation, verbal description, comparison, or a number-line interpretation when supported by the skill), while staying within the selected objectives. Do not repeat the same template with changed numbers only.",
            "seed": batch_seed,
        }
        schema = {"type": "ARRAY", "items": task_schema, "minItems": task_count, "maxItems": task_count}
        body = {
            "contents": [{"parts": [{"text": json.dumps(prompt, ensure_ascii=False)}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema, "temperature": 0.8},
        }
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:streamGenerateContent?alt=sse"
        )
        http_request = Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.gemini_api_key},
            method="POST",
        )
        buffer = ""
        cursor = 0
        emitted = 0
        try:
            with _urlopen_with_retries(http_request, timeout=60) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    fragments = []
                    for candidate in chunk.get("candidates", []):
                        for part in candidate.get("content", {}).get("parts", []):
                            if isinstance(part.get("text"), str):
                                fragments.append(part["text"])
                    buffer += "".join(fragments)
                    decoded, cursor, _ = _extract_json_array_items(buffer, cursor)
                    for result in decoded:
                        if emitted >= task_count:
                            raise TaskGenerationError("gemini_generation_failed", "Gemini returned more tasks than requested.")
                        yield self._build_gemini_task(request, result, batch_seed + emitted)
                        emitted += 1
            if emitted != task_count:
                missing_count = task_count - emitted
                if missing_count > 0 and _partial_attempt < 2:
                    retry_item = request.plan_item.model_copy(
                        update={
                            "task_allocations": [
                                TaskAllocation(
                                    task_type="closed_single_choice",
                                    count=missing_count,
                                )
                            ]
                        }
                    )
                    retry_request = request.model_copy(
                        update={"plan_item": retry_item}
                    )
                    yield from self.generate_batch_stream(
                        retry_request,
                        seed=batch_seed + 1_000_003 * (_partial_attempt + 1),
                        _partial_attempt=_partial_attempt + 1,
                    )
                    return
                raise TaskGenerationError(
                    "gemini_generation_failed",
                    f"Gemini streamed {emitted} complete tasks; the plan requested {task_count}.",
                )
        except TaskGenerationError:
            raise
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as error:
            details = error.read().decode("utf-8", errors="replace") if isinstance(error, HTTPError) else str(error)
            raise TaskGenerationError(
                "gemini_generation_failed",
                f"Gemini streaming request for model '{self.gemini_model}' failed: {details}",
            ) from error

    def _build_gemini_task(self, request, result, seed):
        try:
            statement = result["statement"].strip()
            correct = result["correct_answer"].strip()
            distractors = [value.strip() for value in result["distractors"]]
            steps = result["solution_steps"]
            explanation = result["explanation"].strip()
        except (KeyError, TypeError, AttributeError) as error:
            raise TaskGenerationError("gemini_format_failed", f"Incomplete task fields: {error}") from error
        if (
            not statement
            or not is_statement_well_formed(statement)
            or not correct
            or any(not value for value in distractors)
            or len({value.casefold() for value in [correct, *distractors]}) < 4
            or not isinstance(steps, list)
            or not steps
            or any(not isinstance(step, str) or not step.strip() for step in steps)
            or not explanation
        ):
            raise TaskGenerationError(
                "gemini_format_failed",
                "Task failed local format checks (required fields, three unique distractors, or solution steps).",
            )
        unique_distractors = []
        seen_answers = {correct.casefold()}
        for distractor in distractors:
            normalized = distractor.casefold()
            if normalized not in seen_answers:
                seen_answers.add(normalized)
                unique_distractors.append(distractor)
        distractors = unique_distractors[:3]
        answers = self._build_answers([correct], distractors)
        correct_ids = [answer.id for answer in answers if answer.text == correct]
        if len(correct_ids) != 1:
            raise TaskGenerationError("gemini_format_failed", "Expected exactly one correct answer.")
        comparison_valid = has_single_true_comparison(
            [answer.text for answer in answers], [correct]
        )
        if comparison_valid is False:
            raise TaskGenerationError(
                "gemini_format_failed",
                "Comparison task has multiple true options or an incorrect answer key.",
            )
        item = request.plan_item
        return TaskTemplate(
            task_id=str(uuid4()),
            plan_id=request.plan_id,
            skill_id=item.skill_id,
            skill_ids=item.skill_ids,
            subskill_ids=item.subskill_ids,
            subskills_by_skill=item.subskills_by_skill,
            difficulty=int(round(item.target_difficulty)),
            requested_difficulty=item.target_difficulty,
            curriculum_context=request.curriculum_context,
            material_scope=item.material_scope,
            statement=statement,
            answers=answers,
            correct_answers=correct_ids,
            solution_steps=steps,
            explanation=explanation,
            generation_seed=seed,
            validation_checks=[
                "skill_exists", "curriculum_context", "answer_pool_unique",
                "correct_answer_in_pool", "distractor_count", "gemini_self_check_prompt",
            ],
            generator_version=GENERATOR_VERSION,
        )

    def _generate_with_gemini(self, request, seed):
        """Generate content from the selected skill record, then validate locally."""
        skill_ids = request.plan_item.skill_ids or [request.plan_item.skill_id]
        skills = [self.skills[value] for value in skill_ids]
        schema = {
            "type": "OBJECT",
            "properties": {
                "statement": {"type": "STRING"},
                "correct_answer": {"type": "STRING"},
                "distractors": {"type": "ARRAY", "items": {"type": "STRING"}},
                "solution_steps": {"type": "ARRAY", "items": {"type": "STRING"}},
                "explanation": {"type": "STRING"},
            },
            "required": ["statement", "correct_answer", "distractors", "solution_steps", "explanation"],
        }
        prompt = {
            "instruction": (
                "Ułóż jedno krótkie, jednoznaczne zadanie zamknięte po polsku. Docelową klasę "
                "odczytaj z zakresu materiału w planie; nie obniżaj poziomu tylko dlatego, że "
                "skill wprowadzono wcześniej. Treść zadania napisz jako jedno bezpośrednie polecenie, "
                "najlepiej 5-15 słów. Bez historyjki, opisu sytuacji życiowej, wstępu, zbędnych "
                "danych ani kilku pytań. Nie proś o uzasadnienie w treści zadania. Dobierz liczby "
                "i operacje do celu i trudności z planu; nie zwiększaj trudności przez długi opis. "
                "Stosuj wytyczne skillu i programu. Przed odpowiedzią samodzielnie rozwiąż zadanie "
                "i sprawdź wynik oraz błędność dystraktorów. Zwróć wyłącznie gotową treść zadania; "
                "nie pokazuj wersji roboczych, autokorekty ani tekstów typu 'Nie, wskaż...'. "
                "W porównaniach sprawdź każdy wariant "
                "i nie używaj relacji równoważnych poprawnej odpowiedzi (np. >= jako dystraktora dla >). "
                "Zwróć dokładnie trzy różne dystraktory i "
                "sprawdzalne kroki rozwiązania. Przykładowa forma: "
                "Zamień podany ułamek na liczbę mieszaną. Dobierz konkretne dane do skillu."
            ),
            "skill_guidelines": skills,
            "primary_skill_id": request.plan_item.skill_id,
            "integrated_skills": skill_ids,
            "target_subskills": request.plan_item.subskill_ids,
            "subskills_by_skill": request.plan_item.subskills_by_skill,
            "integration_instruction": "If multiple skills are listed, create a task that requires their combined use. Target only the listed subskills for each skill.",
            "plan": request.plan_item.model_dump(mode="json"),
            "curriculum_context": request.curriculum_context.model_dump(mode="json"),
            "difficulty": round(request.plan_item.target_difficulty),
            "seed": seed,
        }
        body = {
            "contents": [{"parts": [{"text": json.dumps(prompt, ensure_ascii=False)}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema,
                "temperature": 0.8,
            },
        }
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
        http_request = Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.gemini_api_key},
            method="POST",
        )
        try:
            with _urlopen_with_retries(http_request, timeout=45) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            text = response_data["candidates"][0]["content"]["parts"][0]["text"]
            result = json.loads(text)
            statement = result["statement"].strip()
            correct = [result["correct_answer"].strip()]
            distractors = [value.strip() for value in result["distractors"]]
            steps = result["solution_steps"]
            explanation = result["explanation"].strip()
            unique_distractors = []
            seen_answers = {correct[0].casefold()}
            for distractor in distractors:
                if distractor.casefold() not in seen_answers:
                    seen_answers.add(distractor.casefold())
                    unique_distractors.append(distractor)
            distractors = unique_distractors[:3]
            if (len(distractors) != 3 or not statement or not is_statement_well_formed(statement)
                    or not steps or not explanation):
                raise ValueError("Gemini returned incomplete or duplicate task content")
            comparison_valid = has_single_true_comparison(
                [correct[0], *distractors], correct
            )
            if comparison_valid is False:
                raise ValueError(
                    "Gemini returned multiple true comparisons or an incorrect answer"
                )
            return statement, correct, distractors, steps, explanation
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as error:
            details = error.read().decode("utf-8", errors="replace") if isinstance(error, HTTPError) else str(error)
            raise TaskGenerationError(
                "gemini_generation_failed",
                f"Gemini request for model '{self.gemini_model}' failed: {details}. "
                "Check GEMINI_MODEL, GEMINI_API_KEY, and model access.",
            ) from error

    def _validate_context(self, skill_id: str, context: CurriculumContext) -> None:
        curriculum = self.skills[skill_id].get("curriculum", {})
        introduced_grade = curriculum.get("introduced_in_grade")
        if introduced_grade is not None and context.grade < introduced_grade:
            raise TaskGenerationError(
                "outside_curriculum_context",
                f"Skill '{skill_id}' is introduced in grade {introduced_grade}.",
            )
        skill_level = _normalize(curriculum.get("level", ""))
        requested_level = _normalize(context.curriculum_level)
        if (
            skill_level in {"extension", "advanced", "rozszerzenie"}
            and requested_level in {"standard", "basic", "podstawa"}
        ):
            raise TaskGenerationError(
                "outside_curriculum_context",
                f"Skill '{skill_id}' requires an extended curriculum level.",
            )
        school_type = curriculum.get("school_type")
        if school_type and _normalize(school_type) not in _normalize(
            context.education_stage
        ):
            if not (
                school_type == "PRIMARY"
                and any(
                    marker in _normalize(context.education_stage)
                    for marker in ("primary", "podstaw", "elementary")
                )
            ):
                raise TaskGenerationError(
                    "outside_curriculum_context",
                    f"Skill '{skill_id}' is not in the selected education stage.",
                )

    def _validate_plan_subskills(self, plan_item) -> None:
        for skill_id, subskill_ids in plan_item.subskills_by_skill.items():
            valid_ids = {
                objective.get("id")
                for objective in self.skills[skill_id].get("objectives", [])
            }
            unknown = set(subskill_ids) - valid_ids
            if unknown:
                raise TaskGenerationError(
                    "unknown_subskill",
                    f"Unknown objectives for skill '{skill_id}': {', '.join(sorted(unknown))}",
                )

    @staticmethod
    def _build_answers(correct, distractors):
        correct_set = set(correct)
        unique_distractors = []
        for value in distractors:
            text = str(value)
            if text not in correct_set and text not in unique_distractors:
                unique_distractors.append(text)
        if len(unique_distractors) < 2:
            raise TaskGenerationError(
                "insufficient_distractors",
                "Could not construct two unique incorrect options.",
            )
        options = [str(value) for value in correct] + unique_distractors
        return [
            AnswerOption(id=f"answer-{index}", text=value)
            for index, value in enumerate(options, start=1)
        ]


class TaskPresentationService:
    """Create randomized learner views while keeping the answer key server-side."""

    def __init__(self, rng=None, key_store: PresentationKeyStore | None = None):
        self.rng = rng or random.SystemRandom()
        self.key_store = key_store or InMemoryPresentationKeyStore()

    def present(
        self,
        task: TaskTemplate,
        distractor_count: int | None = None,
    ) -> PresentedTask:
        correct_source_id = self.rng.choice(task.correct_answers)
        distractors = [
            option
            for option in task.answers
            if option.id not in task.correct_answers
        ]
        if distractor_count is None:
            distractor_count = min(3, len(distractors))
        if distractor_count > len(distractors):
            raise TaskGenerationError(
                "insufficient_distractors",
                "Not enough distractors for the requested presentation.",
            )
        selected = self.rng.sample(distractors, distractor_count)
        selected.append(
            next(option for option in task.answers if option.id == correct_source_id)
        )
        self.rng.shuffle(selected)

        public_options = [
            PresentedOption(id=str(uuid4()), text=option.text)
            for option in selected
        ]
        correct_public_id = next(
            public.id
            for public, source in zip(public_options, selected)
            if source.id == correct_source_id
        )
        presentation_id = str(uuid4())
        self.key_store.save(
            presentation_id,
            correct_public_id,
            {
                public.id: source.id
                for public, source in zip(public_options, selected)
            },
            task_id=task.task_id,
            presented_task={
                "presentation_id": presentation_id,
                "plan_id": task.plan_id,
                "task_id": task.task_id,
                "task_type": task.task_type,
                "skill_id": task.skill_id,
                "skill_ids": task.skill_ids,
                "subskill_ids": task.subskill_ids,
                "subskills_by_skill": task.subskills_by_skill,
                "difficulty": task.difficulty,
                "statement": task.statement,
                "options": [option.model_dump(mode="json") for option in public_options],
            },
        )

        return PresentedTask(
            presentation_id=presentation_id,
            plan_id=task.plan_id,
            task_id=task.task_id,
            task_type=task.task_type,
            skill_id=task.skill_id,
            skill_ids=task.skill_ids,
            subskill_ids=task.subskill_ids,
            subskills_by_skill=task.subskills_by_skill,
            difficulty=task.difficulty,
            statement=task.statement,
            options=public_options,
        )

    def grade(self, presentation_id: str, selected_option_id: str):
        from task_generator.models import TaskResult

        key = self.key_store.get(presentation_id)
        if key is None:
            raise TaskGenerationError("unknown_presentation", "Presentation not found.")
        if selected_option_id not in key["option_source_ids"]:
            raise TaskGenerationError(
                "invalid_answer", "Selected option was not shown for this presentation."
            )
        correct = selected_option_id == key["correct_option_id"]
        return TaskResult(
            presentation_id=presentation_id,
            selected_option_id=selected_option_id,
            correct=correct,
            score=1.0 if correct else 0.0,
        )


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).strip()
