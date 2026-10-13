"""Deterministic conversational framing, with facts rendered separately."""
from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    description: str
    participants: tuple[str, str]
    opening: str
    closing: str
    lead_ins: tuple[str, ...]


COMMON_SCENARIOS = (
    Scenario("archive_review", "colleagues reviewing historical archive entries", ("Marco", "Elena"),
             "Let's compare the entries in these folders.", "I'll put the folders back in order.",
             ("This entry reads:", "The next note says:", "Here's another entry:")),
    Scenario("study_group", "students comparing notes during a study group", ("Aisha", "Tom"),
             "Shall we go through our revision notes together?", "That gives us something to review before class.",
             ("I have this in my notes:", "Here's another point for revision:", "My notes also say:")),
    Scenario("editorial_check", "editors checking background facts for an article", ("Maya", "Noah"),
             "Let's check the background notes before we edit the draft.", "I'll work on the wording of the draft next.",
             ("For the background section:", "The supporting note says:", "Here's an entry for the draft:")),
    Scenario("exhibition_planning", "curators preparing captions for a historical exhibition", ("Kenji", "Lina"),
             "We need to decide what belongs in the display captions.", "I'll sketch a layout for the display.",
             ("For a possible caption:", "This entry might fit the display:", "The caption notes include this:")),
    Scenario("quiz_preparation", "friends assembling questions for a quiz", ("Sofia", "Daniel"),
             "Let's collect a few entries for our quiz notes.", "We can decide on the question wording later.",
             ("Here's a note for the quiz:", "Another entry we could use:", "This is in my quiz notes:")),
    Scenario("documentary_research", "researchers organizing background material for a documentary", ("Nina", "Owen"),
             "Let's sort the background material for the project.", "I'll organize these notes with the other material.",
             ("The research notes say:", "For the background material:", "Here's an entry from the research folder:")),
)

DOMAIN_SCENARIOS = {
    "countries_byGDP": (
        Scenario("history_seminar", "seminar partners discussing political history", ("David", "Leila"),
                 "Let's go over the political history notes for our seminar.", "I'll bring the notes to the discussion.",
                 ("The seminar notes include this:", "Another entry for our discussion:", "The history notes say:")),
        Scenario("civics_lesson", "a teacher and colleague preparing a civics lesson", ("Sara", "Ben"),
                 "Let's choose background material for the civics lesson.", "I'll arrange the material for the lesson.",
                 ("For the lesson notes:", "Here's a record for the class discussion:", "The background entry reads:")),
    ),
    "companies_byRevenue": (
        Scenario("business_case", "students preparing a business case discussion", ("Iris", "Leo"),
                 "Let's compare the leadership notes for our business case.", "I'll organize the case discussion outline.",
                 ("The case notes say:", "For the leadership background:", "Here's another case entry:")),
        Scenario("company_history", "researchers assembling company history notes", ("Ruth", "Amir"),
                 "Let's work through the company history folders.", "I'll file the notes with the company histories.",
                 ("The company history entry says:", "Here's a note from the folder:", "For the historical summary:")),
    ),
    "athletes_byPayment": (
        Scenario("sports_commentary", "commentators preparing background notes for a sports discussion", ("Sam", "Eva"),
                 "Let's review the career notes before our sports discussion.", "I'll arrange the notes for the discussion.",
                 ("The career notes say:", "Here's another entry for our discussion:", "For the background notes:")),
        Scenario("career_profile", "writers preparing athlete career profiles", ("Theo", "Mila"),
                 "Let's compare the material for these career profiles.", "I'll outline the profiles using our notes.",
                 ("The profile notes include this:", "Here's another career entry:", "For the profile background:")),
    ),
    "organizations": (
        Scenario("institutional_history", "archivists preparing institutional history notes", ("Grace", "Arun"),
                 "Let's check the folders for the institutional histories.", "I'll organize the entries for the history notes.",
                 ("The institutional record says:", "Here's another archival note:", "For the history notes:")),
        Scenario("governance_seminar", "students preparing a seminar on institutional governance", ("Priya", "Hugo"),
                 "Let's review the institutional background for our seminar.", "I'll arrange the notes for our seminar outline.",
                 ("The seminar background says:", "Here's a record for the discussion:", "Another note for the seminar:")),
    ),
}

PLAIN_SCENARIO = Scenario("plain", "a discussion of historical records", ("Marco", "Elena"), "", "", ())


def scenario_pool(domain_id: str) -> tuple[Scenario, ...]:
    return COMMON_SCENARIOS + DOMAIN_SCENARIOS.get(domain_id, ())


def select_scenario(domain_id: str, stage: int, rng: random.Random) -> Scenario:
    candidates = list(scenario_pool(domain_id))
    rng.shuffle(candidates)
    return candidates[(stage - 1) % len(candidates)]


def fact_statement(domain_id: str, subject: str, relation: str, value: str, endpoint: str, valid_time: str) -> str:
    if endpoint not in {"start", "end"}:
        raise ValueError(f"unsupported endpoint: {endpoint}")
    action = "began" if endpoint == "start" else "ended"
    when = "on" if len(valid_time) == 10 else "in"
    if domain_id == "athletes_byPayment":
        return f"{subject}'s recorded affiliation with {value} {action} {when} {valid_time}."
    if domain_id == "companies_byRevenue":
        return f"{value}'s tenure as chief executive officer of {subject} {action} {when} {valid_time}."
    if domain_id == "organizations":
        if relation.lower() == "headquarters location":
            return f"{subject}'s use of {value} as its headquarters {action} {when} {valid_time}."
        return f"{value}'s term as {relation} of {subject} {action} {when} {valid_time}."
    return f"{value}'s term as {relation} {action} {when} {valid_time}."
