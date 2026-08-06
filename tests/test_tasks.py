"""Task layer, and the flagship job-application task.

The assertions worth making here are about what the task REFUSES to supply.
A form submitted under someone's real name is the wrong place for a helpful
guess — the applicant is the one who has to defend it in an interview,
without knowing it was said on their behalf.
"""
from baton.tasks.base import Task
from baton.tasks.job_apply import JobApplication


APPLICANT = {
    "name": "Bernardo Gavazzi",
    "email": "someone@example.com",
    "location": "Rio de Janeiro, Brazil",
    "github": "https://github.com/BGavazzi",
}


def test_task_renders_only_the_facts_it_was_given():
    task = Task(name="t", goal="g", context={"email": "a@b.c", "phone": ""})
    rendered = task.render_context()
    assert "email: a@b.c" in rendered
    assert "phone" not in rendered


def test_empty_context_renders_as_empty_not_as_a_stray_header():
    assert Task(name="t", goal="g").render_context() == ""


# ── job application ──

def test_supplied_applicant_facts_reach_the_context():
    task = JobApplication(applicant=APPLICANT, job_title="AI Engineer", company="Acme")
    rendered = task.render_context()
    assert "Bernardo Gavazzi" in rendered
    assert "https://github.com/BGavazzi" in rendered


def test_unsupplied_fields_are_absent_rather_than_blank():
    """A blank value in the prompt reads as a field to fill in; an absent one
    reads as unknown."""
    task = JobApplication(applicant=APPLICANT)
    rendered = task.render_context()
    assert "phone" not in rendered
    assert "linkedin" not in rendered


def test_context_states_that_anything_missing_must_be_asked_about():
    rendered = JobApplication(applicant=APPLICANT).render_context()
    assert "UNKNOWN" in rendered
    assert "Do not invent" in rendered


def test_context_forbids_substituting_a_similar_field():
    """The subtle failure: filling "portfolio" with the GitHub URL, or a
    salary expectation with a number from the posting."""
    rendered = JobApplication(applicant=APPLICANT).render_context()
    assert "substitute a similar value" in rendered


def test_banked_answers_are_passed_through_as_the_applicants_own_words():
    task = JobApplication(
        applicant=APPLICANT,
        answers={"Why do you want to work here?": "Because I already build with your stack."},
    )
    rendered = task.render_context()
    assert "Why do you want to work here?" in rendered
    assert "Because I already build with your stack." in rendered


def test_goal_requires_asking_rather_than_guessing_on_free_text_questions():
    goal = JobApplication(applicant=APPLICANT).goal
    assert "ask" in goal
    assert "cover letter" in goal


def test_goal_requires_confirming_submission_actually_succeeded():
    """A form that silently failed validation looks much like one that was
    accepted — declaring success from the click alone is how an agent
    reports applications that never happened."""
    goal = JobApplication(applicant=APPLICANT).goal
    assert "success state" in goal
    assert "Submit only once" in goal


def test_task_name_identifies_the_posting():
    task = JobApplication(applicant=APPLICANT, job_title="AI Engineer", company="Acme")
    assert "AI Engineer" in task.name and "Acme" in task.name


def test_task_name_degrades_gracefully_without_title_or_company():
    assert JobApplication(applicant=APPLICANT).name == "apply: job application"


def test_resume_path_is_surfaced_so_the_agent_can_attach_it():
    task = JobApplication(applicant=APPLICANT, resume_path="/tmp/cv.pdf")
    assert "/tmp/cv.pdf" in task.render_context()


def test_application_gets_a_larger_step_budget_than_the_default():
    """Multi-page application flows routinely exceed the 40-step default."""
    assert JobApplication(applicant=APPLICANT).max_steps == 60
