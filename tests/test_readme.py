import pathlib, re
ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIRED = ["Team details", "College / Incubator information", "Project Title", "Problem Statement", "Healthcare Use Case", "Technical Stack",
            "AI/ML Model or Framework Details", "Demo Video", "Open-source License Details", "Architecture Diagram", "Presentation", "Public accessibility"]
ALLOWED = {"[INSERT YOUTUBE UNLISTED VIDEO LINK HERE]", "[INSERT GOOGLE DRIVE PITCH DECK LINK HERE]"}

def test_mandatory_readme_sections_present():
    r = (ROOT / "README.md").read_text()
    for h in REQUIRED: assert f"### {h}" in r, h

def test_linked_files_exist():
    r = (ROOT / "README.md").read_text()
    for f in ("docs/architecture_diagram.pdf", "presentation/project_presentation.pdf", "LICENSE", "docs/demo_script.md"):
        assert f in r and (ROOT / f).exists(), f
    assert "unlisted" in r.lower() and "[INSERT YOUTUBE UNLISTED VIDEO LINK HERE]" in r

def test_team_details_and_only_allowed_placeholders():
    r = (ROOT / "README.md").read_text()
    for t in ("ZWEIKERN", "Shada Ali Kuzhikattil", "Rushdha V", "Vel Tech Rangarajan", "github.com/Shada0/NephroTwin-AI"): assert t in r, t
    assert set(re.findall(r"\[INSERT[^\]]*\]", r)) <= ALLOWED and "PLACEHOLDER" not in r
