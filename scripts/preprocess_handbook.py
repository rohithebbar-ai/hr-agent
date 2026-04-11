# HR AI Copilot — Updated Preprocessing Script (Step 1) 
# ═══════════════════════════════════════════════════════════
# Usage:
#     uv run python scripts/preprocess_handbook.py


# ═══════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════

import json
import re
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pymupdf
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════

@dataclass
class Page:
    number: int
    text: str
    headings: list[str] = field(default_factory=list)
    blocks: list[dict] = field(default_factory=list)
    tables_markdown: list[str] = field(default_factory=list)


class PolicyObject(BaseModel):
    policy_id: str
    section: str
    policy_name: str
    full_text: str
    page_start: int
    page_end: int
    category: str = ""
    keywords: list[str] = []


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

RAW_DIR = Path("data/hr_documents/raw")
PROCESSED_DIR = Path("data/hr_documents/processed")
PROCESSED_POLICY_AWARE_DIR = Path("data/hr_documents/processed/policy_aware")

HANDBOOK_URL = (
    "https://www.franinsurance.com/media/pbka50b5/"
    "employeehandbookandguidelines.pdf"
)
HANDBOOK_FILENAME = "gallagher_employee_handbook.pdf"
SUPPLEMENTARY_FILE = RAW_DIR / "supplementary_policies.txt"


# ═══════════════════════════════════════════════════════
# PLACEHOLDER REPLACEMENTS
# ═══════════════════════════════════════════════════════

PLACEHOLDER_REPLACEMENTS = {
    # Company identity
    "[ORGANIZATION NAME]": "VanaciPrime",
    "[Organization Name]": "VanaciPrime",
    "[ORGANIZATION]": "VanaciPrime",
    "[Company Name]": "VanaciPrime",
    "[Company]": "VanaciPrime",
    "[company]": "VanaciPrime",
    "[Employer]": "VanaciPrime",
    "[EMPLOYER]": "VanaciPrime",

    # Location
    "[STATE]": "California",
    "[State]": "California",
    "[CITY]": "San Francisco",
    "[City]": "San Francisco",
    "[ADDRESS]": "100 Innovation Drive, San Francisco, CA 94105",

    # Contacts
    "[HR CONTACT]": "hr@vanaciprime.com",
    "[HR Manager]": "the HR Manager",
    "[Phone Number]": "(415) 555-0100",
    "[insert phone number]": "(415) 555-0100",
    "[Email]": "hr@vanaciprime.com",
    "[Website]": "www.vanaciprime.com",

    # Time and hours
    "[NUMBER OF HOURS]": "40",
    "[Number of Hours]": "40",
    "[DAYS]": "90",
    "[Days]": "90",

    # PTO and leave
    "[insert # of hours]": "40",
    "[insert hour]": "9:00 AM",
    "[insert # of months]": "3",
    "[insert year]": "2026",

    # Safety
    "[insert #]": "2",
    "[insert location]": "the north and south ends of the building",

    # Car rental and insurance
    "[insert agency name]": "Enterprise Rent-A-Car",
    "[insert carrier name]": "Hartford Insurance",

    # Health insurance
    "[insert amount] percent discount": "5 percent discount",

    # Bulletin board
    "[insert site]": "https://intranet.vanaciprime.com/bulletin",

    # Remaining placeholders from v1 run
    "[insert plan name]": "VanaciPrime Health Plus",
    "[insert number]": "5",
    "[insert time]": "9:00 AM",
    "[insert number of weeks]": "12",
    "[insert mailing information]": "VanaciPrime HR Department, 100 Innovation Drive, San Francisco, CA 94105",
    "[insert number days]": "30 days",
    "[insert name and phone number]": "HR Department at (415) 555-0100",
    "[insert department of person to\ncontact]": "Human Resources Department",
    "[insert number of days]": "30 days",
    "[insert\nlocation]": "the north and south ends of the building",
    "[insert name]": "VanaciPrime HR",
    "[insert amount of time]": "30 minutes",
    "[insert dollar amount]": "50",
    "[Insert amount here]": "$500",
    "[insert amount]": "$500",
    "[insert number\nhere]": "5",
    "[insert department or person who will handle lost or\nstolen company credit cards here]": "the Finance Department at (415) 555-0102",
    "[insert email address]": "hr@vanaciprime.com",
    "[insert address]": "100 Innovation Drive, San Francisco, CA 94105",
    "[insert external channels for\nescalation here]": "the California Department of Fair Employment and Housing (DFEH)",
    "[insert relevant designations or certifications]": "relevant professional certifications",
    "[insert license type]": "professional license",
    "[insert dollar amount]": "$50",
    "[insert name, title]": "Sarah Chen, VP of Human Resources",
    "[insert time and day of the calendar week]": "5:00 PM on Friday",
    "[insert time and day of\ncalendar week]": "9:00 AM on Monday",
    "[insert number of hours]": "40",
    "[insert day of the week and time of\nday]": "Monday at 9:00 AM",
    "[insert day of the week and time of day]": "Friday at 5:00 PM",
    "[insert payment frequency:\nweekly/semi-monthly/monthly]": "semi-monthly",
    "[insert regularly occurring paydays, such as 15 and\n30]": "15th and last day",
    "[insert validation method,\nsuch as initialing the time card or submitting the record through a digital time tracking system]": "submitting the record through the digital time tracking system",
}


# ═══════════════════════════════════════════════════════
# SENTENCE REPLACEMENTS
# ═══════════════════════════════════════════════════════

SENTENCE_REPLACEMENTS = {
    "Standard working hours are from [insert hour] to [insert hour],"
    " Monday through Friday. A [insert amount of time] lunch"
    " period is taken at any hour":
        "Standard working hours are from 9:00 AM to 5:00 PM,"
        " Monday through Friday. A 1 hour lunch"
        " period is taken at any hour",

    "limits not less than [insert amount] for bodily injury"
    " and [insert amount] for property damage":
        "limits not less than $100,000 for bodily injury"
        " and $50,000 for property damage",

    "For any tickets with a round trip cost over [insert cost]":
        "For any tickets with a round trip cost over $500",
}


# ═══════════════════════════════════════════════════════
# CONTEXT-AWARE REPLACEMENTS (regex-based)
# ═══════════════════════════════════════════════════════
# Replaces placeholders based on their surrounding context.
# Each pattern matches a specific phrase, not just the placeholder.

CONTEXT_AWARE_REPLACEMENTS = [
    # ── Lunch and rest periods ──
    (
        r"\[insert number\]-minute lunch break",
        "30-minute lunch break",
    ),
    (
        r"hours of \[insert time\] and \[insert time\]",
        "hours of 11:30 AM and 1:30 PM",
    ),
    (
        r"Two paid rest periods of \[insert number\] minutes",
        "Two paid rest periods of 15 minutes",
    ),

    # ── Employee classification (full-time / part-time) ──
    (
        r"work at least \[insert number\] hours per week",
        "work at least 40 hours per week",
    ),
    (
        r"work fewer than \[insert number\] hours per week",
        "work fewer than 40 hours per week",
    ),
    (
        r"work \[insert number\] hours or fewer per week",
        "work 20 hours or fewer per week",
    ),

    # ── Salary advance ──
    (
        r"no more than \[insert number\] months of net pay",
        "no more than 2 months of net pay",
    ),
    (
        r"in no more than \[insert number\] equal installments",
        "in no more than 6 equal installments",
    ),
    (
        r"more than \[insert number\] months of repayment",
        "more than 6 months of repayment",
    ),
    (
        r"\[insert number\] percent of the advance",
        "5 percent of the advance",
    ),

    # ── Vacation request form ──
    (
        r"\[insert number\]-hour increments",
        "4-hour increments",
    ),

    # ── Benefits eligibility ──
    (
        r"\[insert # of hours\] hours per week",
        "40 hours per week",
    ),
    (
        r"after \[insert # of months\] months",
        "after 3 months",
    ),

    # ── Time and dollar amounts (specific contexts) ──
    (
        r"deadline of \[insert hour\]",
        "deadline of 9:00 AM",
    ),
    (
        r"by \[insert hour\] on",
        "by 9:00 AM on",
    ),

    # ── Salary advance dollar amount fix ──
    (
        r"processing fee of \$\[insert dollar amount\]",
        "processing fee of $50",
    ),

    # ── Vacation Policy (uses parentheses, not brackets!) ──
    (
        r"full-time employees are permitted \(insert amount here\) paid vacation",
        "full-time employees are permitted 15 days of paid vacation",
    ),
    (
        r"part-time employees will be eligible for \(insert amount here\) of paid vacation",
        "part-time employees will be eligible for 8 days of paid vacation",
    ),
    
    # ── PTO Policy (parenthesis variant) ──
    (
        r"part-time\s*employees earn PTO by working at least \(insert # of hours\) hours",
        "part-time employees earn PTO by working at least 20 hours",
    ),
]


def apply_context_aware_replacements(text: str) -> str:
    """
    Apply position-aware regex replacements BEFORE global placeholder
    replacement. This handles cases where the same placeholder
    (e.g., [insert number]) needs different values in different
    contexts.
    """
    for pattern, replacement in CONTEXT_AWARE_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

# ═══════════════════════════════════════════════════════
# BLANK FIELD PATTERNS (regex)
# ═══════════════════════════════════════════════════════

BLANK_FIELD_PATTERNS = [
    (r"(\d+\s*)?_{3,}\s*(vacation|PTO)\s*(days|hours)",
     "15 vacation days"),
    (r"_{3,}\s*sick\s*(days|leave)",
     "10 sick days"),
    (r"_{3,}\s*personal\s*(days|leave)",
     "3 personal days"),
    (r"probationary\s*period\s*(is|of)\s*_{3,}",
     "probationary period is 90"),
    (r"health\s*insurance\s*begins?\s*(after)?\s*_{3,}",
     "health insurance begins after 30"),
    (r"_{3,}\s*%\s*(of|match)",
     "4% match"),
    (r"overtime\s*(is)?\s*paid\s*at\s*_{3,}",
     "overtime is paid at 1.5"),
    (r"pay\s*period\s*(is)?\s*_{3,}",
     "pay period is bi-weekly"),
]


REPEATED_LINES = [
    "This Employee Handbook policy is a guideline meant to be edited",
    "It is not meant to be exhaustive or construed as legal advice",
    "Consult additional insurance and/or legal counsel",
    "professional advice.",
    "\u00a9 2017-2018 Zywave, Inc.",
    "\u00a9 2017-2018 Zywave",
    "Zywave, Inc. All rights reserved",
    "All rights reserved.",
]

SECTION_CATEGORIES = {
    "Introduction": "General",
    "Employment Policies": "Employment",
    "Employment": "Employment",
    "Termination": "Employment",
    "Transfer": "Employment",
    "Promotion": "Employment",
    "Immigration": "Employment",
    "Disabilities": "Employment",
    "Equal Employment": "Employment",
    "Workplace Conduct": "Workplace Safety",
    "Harassment": "Workplace Safety",
    "Sexual Harassment": "Workplace Safety",
    "Bullying": "Workplace Safety",
    "Violence": "Workplace Safety",
    "Weapons": "Workplace Safety",
    "Drug": "Workplace Safety",
    "Alcohol": "Workplace Safety",
    "Disciplinary": "Workplace Safety",
    "Code of Ethics": "Workplace Safety",
    "Complaint": "Workplace Safety",
    "Diversity": "Workplace Safety",
    "Employee Benefits": "Benefits",
    "Benefits": "Benefits",
    "COBRA": "Benefits",
    "Insurance": "Benefits",
    "Domestic Partnership": "Benefits",
    "Adoption": "Benefits",
    "Time Away": "Leave",
    "Leave": "Leave",
    "FMLA": "Leave",
    "PTO": "Leave",
    "Vacation": "Leave",
    "Sick": "Leave",
    "Personal Leave": "Leave",
    "Bereavement": "Leave",
    "Funeral": "Leave",
    "Jury": "Leave",
    "Military": "Leave",
    "Nursing": "Leave",
    "Parental": "Leave",
    "Pandemic": "Leave",
    "Communicable": "Leave",
    "Religious": "Leave",
    "Lunch": "Leave",
    "Rest Period": "Leave",
    "Safety": "Safety",
    "Emergency": "Safety",
    "Security": "Security",
    "Information": "Security",
    "General Practices": "Operations",
    "Attendance": "Operations",
    "Dress Code": "Operations",
    "Social Networking": "Operations",
    "Educational": "Operations",
    "Business Expense": "Operations",
    "Company Car": "Operations",
    "Company Credit": "Operations",
    "Travel": "Operations",
    "Compensation": "Compensation",
    "Pay": "Compensation",
    "Salary": "Compensation",
    "Overtime": "Compensation",
    "Appendix": "Forms",
    "Application": "Forms",
    "Receipt": "Forms",
    "Certificate": "Forms",
    "Background Check": "Operations",
    "Personnel": "Operations",
    "Phone": "Operations",
    "Physical Examination": "Operations",
    "Severe Weather": "Safety",
    "Anti-discrimination": "Employment",
}

# Known section headers in this handbook (font_size > 16pt)
# These are the broad groupings, NOT individual policy names
KNOWN_SECTIONS = {
    "Introduction",
    "Employment Policies",
    "Workplace Conduct",
    "Employee Benefits",
    "Time Away From Work",
    "General Practices",
    "Information & Office Security",
    "Compensation",
    "Appendix",
}

def fix_parenthesis_placeholders(text: str) -> str:
    """
    Catch placeholders that use parentheses instead of brackets.
    These are leftover template artifacts the bracket-based 
    replacements missed.
    """
    # Match patterns like (insert amount here), (insert # of hours), etc.
    text = re.sub(
        r"\(insert [^)]+\)",
        "[see policy details]",
        text,
        flags=re.IGNORECASE,
    )
    return text

# ═══════════════════════════════════════════════════════
# DOWNLOAD
# ═══════════════════════════════════════════════════════

def download_handbook() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RAW_DIR / HANDBOOK_FILENAME

    if filepath.exists():
        print(f"[OK] Handbook already exists: {filepath}")
        return filepath

    print(f"[DOWNLOAD] Fetching handbook...")
    urllib.request.urlretrieve(HANDBOOK_URL, filepath)
    print(f"[OK] Downloaded to: {filepath}")
    return filepath


def load_supplementary_policies() -> list[PolicyObject]:
    """
    Load manually-written supplementary policies that aren't in
    the Gallagher handbook. Format: sections separated by 
    '=== Policy Name ===' headers.
    """
    if not SUPPLEMENTARY_FILE.exists():
        print(f"[INFO] No supplementary policies file found")
        return []
    
    print(f"[LOAD] Reading supplementary policies...")
    
    with open(SUPPLEMENTARY_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    
    policies = []
    # Split on === Policy Name === headers
    sections = re.split(r"===\s*([^=]+?)\s*===", content)
    
    # sections is like: ['', 'Policy Name 1', 'body 1', 'Policy Name 2', 'body 2', ...]
    for i in range(1, len(sections), 2):
        if i + 1 >= len(sections):
            break
        
        policy_name = sections[i].strip()
        body = sections[i + 1].strip()
        
        # Parse Section: and Category: lines from the body
        section = "Employment Policies"  # default
        category = "Employment"          # default
        body_lines = []
        
        for line in body.split("\n"):
            if line.startswith("Section:"):
                section = line.replace("Section:", "").strip()
            elif line.startswith("Category:"):
                category = line.replace("Category:", "").strip()
            else:
                body_lines.append(line)
        
        full_text = "\n".join(body_lines).strip()
        
        policies.append(PolicyObject(
            policy_id=f"supplementary_{i//2 + 1:03d}",
            section=section,
            policy_name=policy_name,
            full_text=full_text,
            page_start=999,  # marker for supplementary
            page_end=999,
            category=category,
            keywords=extract_keywords(full_text, policy_name),
        ))
    
    print(f"[OK] Loaded {len(policies)} supplementary policies")
    return policies

# ═══════════════════════════════════════════════════════
# DETECT PAGES TO SKIP
# ═══════════════════════════════════════════════════════

def detect_pages_to_skip(doc) -> set:
    skip = set()

    for page in doc:
        text = page.get_text("text").strip()
        images = page.get_images()
        char_count = len(text)

        # Cover pages
        if char_count < 100 and len(images) > 0:
            skip.add(page.number)
            print(f"  [SKIP] Page {page.number}: Cover page "
                  f"({char_count} chars, {len(images)} images)")
            continue

        # Table of contents
        lines = text.split("\n")
        toc_pattern_count = sum(
            1 for line in lines
            if re.search(r"\.{4,}\s*\d+\s*$", line)
        )
        if toc_pattern_count > 5:
            skip.add(page.number)
            print(f"  [SKIP] Page {page.number}: Table of contents "
                  f"({toc_pattern_count} TOC lines)")
            continue

        # Nearly empty pages
        if char_count < 50:
            skip.add(page.number)
            print(f"  [SKIP] Page {page.number}: Nearly empty")
            continue

        # Signature pages
        text_lower = text.lower()
        if ("signature" in text_lower
                and "date" in text_lower
                and char_count < 300):
            skip.add(page.number)
            print(f"  [SKIP] Page {page.number}: Signature page")
            continue

        # Blank form pages
        tables = page.find_tables()
        if tables and char_count < 200:
            for table in tables:
                data = table.extract()
                total_cells = sum(len(row) for row in data)
                empty_cells = sum(
                    1 for row in data
                    for cell_val in row
                    if not cell_val or cell_val.strip() == ""
                )
                if total_cells > 0 and empty_cells / total_cells > 0.7:
                    skip.add(page.number)
                    print(f"  [SKIP] Page {page.number}: Empty form")
                    break

    print(f"\n[OK] Auto-detected {len(skip)} pages to skip")
    return skip


# ═══════════════════════════════════════════════════════
# EXTRACT WITH PYMUPDF
# ═══════════════════════════════════════════════════════

def extract_with_pymupdf(filepath: Path) -> list[Page]:
    print(f"[EXTRACT] Opening PDF: {filepath}")
    doc = pymupdf.open(str(filepath))
    print(f"[OK] PDF has {len(doc)} pages")

    print("[DETECT] Scanning for pages to skip...")
    skip_pages = detect_pages_to_skip(doc)

    pages: list[Page] = []

    for page in doc:
        if page.number in skip_pages:
            continue

        blocks = page.get_text("dict", sort=True)["blocks"]

        page_text_parts = []
        page_headings = []
        page_blocks = []

        for block in blocks:
            if block["type"] != 0:
                continue

            for line in block.get("lines", []):
                line_text = ""
                max_font_size = 0

                for span in line.get("spans", []):
                    line_text += span["text"]
                    max_font_size = max(max_font_size, span["size"])

                line_text = line_text.strip()
                if not line_text:
                    continue

                page_text_parts.append(line_text)
                page_blocks.append({
                    "text": line_text,
                    "font_size": max_font_size,
                })

                if max_font_size > 13 and len(line_text) > 3:
                    page_headings.append(line_text)

        # Extract tables as markdown
        tables_md = []
        found_tables = page.find_tables()
        for table in found_tables:
            try:
                md = table.to_markdown()
                if md and len(md.strip()) > 10:
                    tables_md.append(md)
            except Exception:
                pass

        full_text = "\n".join(page_text_parts)
        if tables_md:
            full_text += "\n\n" + "\n\n".join(tables_md)

        pages.append(Page(
            number=page.number,
            text=full_text,
            headings=page_headings,
            blocks=page_blocks,
            tables_markdown=tables_md,
        ))

    doc.close()
    print(f"[OK] Extracted {len(pages)} pages "
          f"(skipped {len(skip_pages)})")
    return pages


# ═══════════════════════════════════════════════════════
# TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════

def remove_repeated_lines(text: str) -> str:
    """FIX #1: Properly removes Zywave footer and all repeated lines."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        is_repeated = any(
            pattern.lower() in line.lower()
            for pattern in REPEATED_LINES
        )
        if not is_repeated:
            cleaned.append(line)
    return "\n".join(cleaned)


def replace_sentences(text: str) -> str:
    for original, replacement in SENTENCE_REPLACEMENTS.items():
        text = text.replace(original, replacement)
    return text


def replace_placeholders(text: str) -> str:
    for placeholder, replacement in PLACEHOLDER_REPLACEMENTS.items():
        text = text.replace(placeholder, replacement)
    return text


def fill_blank_fields(text: str) -> str:
    for pattern, replacement in BLANK_FIELD_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"_{5,}", "[see policy details]", text)
    return text


def fix_blank_company_names(text: str) -> str:
    """
    FIX #3: The PDF template has blank spaces where company name
    should be. After bracket replacements, we still have:
    'employment with .' and 'initiated by ' with nothing.
    """
    # "with ." at word boundary → "with VanaciPrime."
    text = re.sub(r"\bwith \.", "with VanaciPrime.", text)

    # "by " followed by newline → "by VanaciPrime"
    text = re.sub(r"\bby \n", "by VanaciPrime\n", text)

    # " 's " (possessive with no name) → "VanaciPrime's "
    text = re.sub(r" 's ", " VanaciPrime's ", text)

    # "of  " or "at  " (double space where name should be)
    text = re.sub(r"\bof  ", "of VanaciPrime ", text)
    text = re.sub(r"\bat  ", "at VanaciPrime ", text)
    text = re.sub(r"\bto  ", "to VanaciPrime ", text)
    text = re.sub(r"\bfor  ", "for VanaciPrime ", text)
    text = re.sub(r"\bfrom  ", "from VanaciPrime ", text)

    # Standalone double space in middle of sentence
    text = re.sub(r"(\w) {2,}(\w)", r"\1 VanaciPrime \2", text)

    return text


def clean_text(text: str) -> str:
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove page number artifacts
    text = re.sub(
        r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE
    )

    # Remove standalone page numbers on their own line
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

    # FIX #2: Remove stray page numbers at end of text
    text = re.sub(r"\n\d{1,3}\s*$", "", text)

    # Strip each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def preprocess_pages(pages: list[Page]) -> list[Page]:
    """
    Full pipeline. Order matters:
    1. Remove repeated headers/footers (Zywave)
    2. Apply CONTEXT-AWARE replacements
    3. Full sentence replacements
    4. Individual placeholder replacements (catches remaining generics)
    5. Fill blank fields (regex)
    6. Fix blank company names
    7. Clean text
    """
    processed: list[Page] = []

    for page in pages:
        text = page.text

        text = remove_repeated_lines(text)
        text = apply_context_aware_replacements(text) 
        text = replace_sentences(text)
        text = replace_placeholders(text)
        text = fix_parenthesis_placeholders(text) 
        text = fill_blank_fields(text)
        text = fix_blank_company_names(text)
        text = clean_text(text)

        if len(text.strip()) < 50:
            continue

        processed.append(Page(
            number=page.number,
            text=text,
            headings=page.headings,
            blocks=page.blocks,
            tables_markdown=page.tables_markdown,
        ))

    print(f"[OK] Preprocessed {len(processed)} pages "
          f"(dropped {len(pages) - len(processed)} empty)")
    return processed


# ═══════════════════════════════════════════════════════
# EXTRACT STRUCTURED POLICY OBJECTS
# FIX #4: Section and policy_name correctly assigned
# FIX #5: Categories properly mapped
# ═══════════════════════════════════════════════════════

def categorize_policy(section: str, policy_name: str) -> str:
    """FIX #5: Check both section AND policy_name for category match."""
    combined = f"{section} {policy_name}"
    for keyword, category in SECTION_CATEGORIES.items():
        if keyword.lower() in combined.lower():
            return category
    return "General"


def extract_keywords(text: str, policy_name: str) -> list[str]:
    name_words = [
        w.lower() for w in policy_name.split()
        if len(w) > 3 and w.lower() not in (
            "policy", "the", "and", "for", "with"
        )
    ]

    hr_keywords = [
        "leave", "fmla", "pto", "vacation", "sick",
        "harassment", "discrimination", "safety",
        "termination", "resignation", "overtime",
        "insurance", "benefits", "cobra", "401k",
        "drug", "alcohol", "testing", "disability",
        "pregnancy", "maternity", "paternity",
        "bereavement", "jury", "military",
        "compensation", "salary", "wage",
    ]

    text_lower = text.lower()
    found = [kw for kw in hr_keywords if kw in text_lower]
    all_keywords = list(set(name_words + found))
    return sorted(all_keywords)[:10]


def extract_policies(pages: list[Page]) -> list[PolicyObject]:
    policies: list[PolicyObject] = []
    current_section = "General"
    current_policy_name = ""
    current_text_parts: list[str] = []
    current_page_start = 0
    current_page_end = 0
    policy_counter = 0

    for page in pages:
        # Collect heading texts from blocks (for font detection)
        heading_texts = set()
        for block in page.blocks:
            if block["font_size"] > 13 and len(block["text"].strip()) > 3:
                heading_texts.add(block["text"].strip())

        # Process headings for section/policy routing
        for heading in page.headings:
            heading = heading.strip()
            if not heading:
                continue

            # Handle em-dash headings
            if "\u2014" in heading:
                parts = heading.split("\u2014", 1)
                current_section = parts[0].strip()
                policy_heading = parts[1].strip()
            elif heading in KNOWN_SECTIONS:
                current_section = heading
                continue
            else:
                policy_heading = heading

            # Save previous policy
            if current_policy_name and current_text_parts:
                policy_counter += 1
                full_text = "\n".join(current_text_parts).strip()

                if "Table of Contents" not in current_policy_name:
                    policies.append(PolicyObject(
                        policy_id=f"policy_{policy_counter:03d}",
                        section=current_section,
                        policy_name=current_policy_name,
                        full_text=full_text,
                        page_start=current_page_start,
                        page_end=current_page_end,
                        category=categorize_policy(
                            current_section, current_policy_name
                        ),
                        keywords=extract_keywords(
                            full_text, current_policy_name
                        ),
                    ))

            current_policy_name = policy_heading
            current_text_parts = []
            current_page_start = page.number
            current_page_end = page.number

        # Remove heading lines from page text so only body remains
        body_lines = []
        for line in page.text.split("\n"):
            line_stripped = line.strip()
            if line_stripped and line_stripped not in heading_texts:
                body_lines.append(line_stripped)

        if body_lines:
            current_text_parts.append("\n".join(body_lines))
            current_page_end = page.number

        # Append tables
        for table_md in page.tables_markdown:
            current_text_parts.append(table_md)

    # Last policy
    if current_policy_name and current_text_parts:
        policy_counter += 1
        full_text = "\n".join(current_text_parts).strip()
        if "Table of Contents" not in current_policy_name:
            policies.append(PolicyObject(
                policy_id=f"policy_{policy_counter:03d}",
                section=current_section,
                policy_name=current_policy_name,
                full_text=full_text,
                page_start=current_page_start,
                page_end=current_page_end,
                category=categorize_policy(
                    current_section, current_policy_name
                ),
                keywords=extract_keywords(
                    full_text, current_policy_name
                ),
            ))

    print(f"[OK] Extracted {len(policies)} policy objects")
    return policies


# ═══════════════════════════════════════════════════════
# SAVE OUTPUTS
# ═══════════════════════════════════════════════════════

def save_policies_json(policies: list[PolicyObject]) -> Path:
    data = [policy.model_dump() for policy in policies]

    # Naive baseline path
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "policies.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {len(policies)} policies to: {output_path}")

    # Policy-aware path (same source data, different chunking strategy in ingest)
    PROCESSED_POLICY_AWARE_DIR.mkdir(parents=True, exist_ok=True)
    policy_aware_path = PROCESSED_POLICY_AWARE_DIR / "policies.json"
    with open(policy_aware_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {len(policies)} policies to: {policy_aware_path}")

    return output_path


def save_section_metadata(policies: list[PolicyObject]) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "section_metadata.json"

    metadata = [
        {
            "policy_id": p.policy_id,
            "section": p.section,
            "policy_name": p.policy_name,
            "category": p.category,
            "keywords": p.keywords,
            "page_start": p.page_start,
            "page_end": p.page_end,
        }
        for p in policies
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved section metadata to: {output_path}")
    return output_path


def save_debug_txt(pages: list[Page]) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "full_text_debug.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(f"\n{'='*80}\n")
            f.write(f"PAGE {page.number}\n")
            if page.headings:
                f.write(
                    f"HEADINGS: {', '.join(page.headings)}\n"
                )
            f.write(f"{'='*80}\n\n")
            f.write(page.text)
            f.write("\n")

    print(f"[OK] Saved debug text to: {output_path}")
    return output_path


def save_remaining_placeholders(pages: list[Page]) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "remaining_placeholders.txt"

    pattern = re.compile(r"\[insert[^\]]*\]", re.IGNORECASE)
    found: list[dict] = []

    for page in pages:
        matches = pattern.findall(page.text)
        for match in matches:
            found.append({
                "page": page.number,
                "placeholder": match,
            })

    with open(output_path, "w", encoding="utf-8") as f:
        if found:
            f.write(f"Found {len(found)} remaining placeholders:\n\n")
            for item in found:
                f.write(
                    f"  Page {item['page']}: {item['placeholder']}\n"
                )
        else:
            f.write("No remaining placeholders found. All replaced!\n")

    status = f"{len(found)} remaining" if found else "All replaced!"
    print(f"[OK] Placeholder check: {status}")
    if found:
        print(f"     Review: {output_path}")

    return output_path


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  STEP 1: Document Preprocessing (v2)")
    print("  VanaciPrime HR Handbook")
    print("  PyMuPDF + Policy Object Architecture")
    print("=" * 60 + "\n")

    # 1. Download
    pdf_path = download_handbook()

    # 2. Extract
    pages = extract_with_pymupdf(pdf_path)

    # 3. Preprocess
    processed_pages = preprocess_pages(pages)

    # 4. Extract policy objects
    policies = extract_policies(processed_pages)

    # 5. Add supplementary policies
    supplementary = load_supplementary_policies()
    policies.extend(supplementary)

    print(f"[OK] Total policies: {len(policies)} "
          f"({len(supplementary)} supplementary)")

    # 6. Save
    print("\n[SAVE] Writing outputs")
    save_policies_json(policies)
    save_section_metadata(policies)
    save_debug_txt(processed_pages)
    save_remaining_placeholders(processed_pages)

    # 7. Summary
    total_chars = sum(len(p.full_text) for p in policies)
    sections = set(p.section for p in policies)
    categories = set(p.category for p in policies)

    print(f"\n{'─'*50}")
    print(f"  SUMMARY")
    print(f"{'─'*50}")
    print(f"  Pages extracted:   {len(pages)}")
    print(f"  Pages after clean: {len(processed_pages)}")
    print(f"  Policy objects:    {len(policies)}")
    print(f"  Total characters:  {total_chars:,}")
    print(f"  Unique sections:   {len(sections)}")
    print(f"  Categories:        {', '.join(sorted(categories))}")
    print(f"{'─'*50}")

    # Print category breakdown
    from collections import Counter
    cat_counts = Counter(p.category for p in policies)
    print(f"\n  Category breakdown:")
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat:20s}: {count}")

    return policies


if __name__ == "__main__":
    main()