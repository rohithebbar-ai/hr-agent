"""
Golden Test Set
───────────────────────
60 Q&A pairs for evaluating RAG quality.
Answers are based on the values we filled during preprocessing.

Usage:
    uv run python scripts/golden_test_set.py
"""

import json
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path("data/golden_test_set")

GOLDEN_TEST_SET = [
    # ══════════════════════════════════════════
    # CATEGORY 1: FACTUAL (15 questions)
    # ══════════════════════════════════════════
    {
        "question": "How many vacation days do full-time employees get per year?",
        "expected_answer": "Full-time employees receive 15 vacation days per year.",
        "source_section": "Time Away From Work",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "question": "How many sick leave days are allowed per year?",
        "expected_answer": "Employees receive 10 sick days per year.",
        "source_section": "Time Away From Work",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "question": "What is the probationary period for new employees?",
        "expected_answer": "The probationary period for new employees is 90 days.",
        "source_section": "Employment Policies",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "question": "At what rate is overtime paid?",
        "expected_answer": "Overtime is paid at 1.5 times the regular hourly rate.",
        "source_section": "Compensation",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "question": "When does health insurance coverage begin for new employees?",
        "expected_answer": "Health insurance begins after 30 days of employment.",
        "source_section": "Employee Benefits",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "question": "What is VanaciPrime's policy on at-will employment?",
        "expected_answer": "VanaciPrime operates under at-will employment. Either the employer or employee can terminate the relationship at any time, with or without cause.",
        "source_section": "Introduction",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "question": "What are the standard working hours at VanaciPrime?",
        "expected_answer": "Standard working hours are from 9:00 AM to 5:00 PM, Monday through Friday.",
        "source_section": "General Practices",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "question": "How often are employees paid?",
        "expected_answer": "Employees are paid on a semi-monthly basis, on the 15th and last day of each month.",
        "source_section": "Compensation",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "question": "Does VanaciPrime conduct drug testing?",
        "expected_answer": "Yes, VanaciPrime reserves the right to conduct drug and alcohol testing including pre-employment, for-cause, and random testing.",
        "source_section": "Workplace Conduct",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "question": "How many personal days do employees receive?",
        "expected_answer": "Employees receive 3 personal days per year.",
        "source_section": "Time Away From Work",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "question": "What is the lunch break duration?",
        "expected_answer": "Employees receive a 1 hour lunch period.",
        "source_section": "General Practices",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "question": "Is VanaciPrime an equal opportunity employer?",
        "expected_answer": "Yes, VanaciPrime is an equal opportunity employer and does not discriminate based on religion, age, gender, national origin, sexual orientation, race or color.",
        "source_section": "Employment Policies",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "question": "What is VanaciPrime's stance on workplace violence?",
        "expected_answer": "VanaciPrime has zero tolerance for workplace violence. All threats or acts of violence are grounds for immediate disciplinary action up to and including termination.",
        "source_section": "Workplace Conduct",
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "question": "What is the preferred car rental agency for business travel?",
        "expected_answer": "VanaciPrime has a preferred relationship with Enterprise Rent-A-Car offering discounted rates and direct billing.",
        "source_section": "General Practices",
        "category": "factual",
        "difficulty": "medium",
    },
    {
        "question": "What is the standard workweek in hours?",
        "expected_answer": "The standard workweek at VanaciPrime consists of 40 hours.",
        "source_section": "Compensation",
        "category": "factual",
        "difficulty": "easy",
    },

    # ══════════════════════════════════════════
    # CATEGORY 2: PROCEDURAL (12 questions)
    # ══════════════════════════════════════════
    {
        "question": "What are the steps to file a harassment complaint?",
        "expected_answer": "Report the incident to your supervisor or HR immediately. Provide written details of the incident. The company will conduct a thorough investigation. Appropriate action will be taken based on findings.",
        "source_section": "Workplace Conduct",
        "category": "procedural",
        "difficulty": "medium",
    },
    {
        "question": "How do I submit a resignation?",
        "expected_answer": "Submit a written resignation to your manager. Provide at least two weeks advance notice. A meeting will take place prior to your last day. Return all company property including parking passes, keys, and equipment.",
        "source_section": "Employment Policies",
        "category": "procedural",
        "difficulty": "medium",
    },
    {
        "question": "What is the process for reporting a workplace injury?",
        "expected_answer": "Report any workplace injury immediately to your supervisor. Complete an incident report. Seek medical attention if needed. The company will file a workers compensation claim.",
        "source_section": "General Practices",
        "category": "procedural",
        "difficulty": "medium",
    },
    {
        "question": "How do I request FMLA leave?",
        "expected_answer": "Provide 30 days advance notice when foreseeable. Submit a request to HR with medical certification. FMLA provides up to 12 weeks of unpaid leave for qualifying events.",
        "source_section": "Time Away From Work",
        "category": "procedural",
        "difficulty": "medium",
    },
    {
        "question": "What should I do if I suspect a coworker is under the influence at work?",
        "expected_answer": "Report your concerns to your supervisor or HR immediately. Do not confront the employee directly. The company will investigate and may require the employee to submit to testing.",
        "source_section": "Workplace Conduct",
        "category": "procedural",
        "difficulty": "medium",
    },
    {
        "question": "How do I request time off?",
        "expected_answer": "Submit a time-off request to your supervisor with sufficient advance notice. Requests are approved based on business needs and staffing requirements.",
        "source_section": "Time Away From Work",
        "category": "procedural",
        "difficulty": "easy",
    },
    {
        "question": "What is the process for an internal transfer or promotion?",
        "expected_answer": "Employees interested in internal opportunities should discuss with their manager and HR. Transfers and promotions are based on qualifications, performance, and business needs.",
        "source_section": "Employment Policies",
        "category": "procedural",
        "difficulty": "medium",
    },
    {
        "question": "How do I report an ethics violation?",
        "expected_answer": "Report ethics violations to your supervisor, HR, or through the company's designated reporting channels. VanaciPrime prohibits retaliation against employees who report violations in good faith.",
        "source_section": "Workplace Conduct",
        "category": "procedural",
        "difficulty": "medium",
    },
    {
        "question": "What should I do if I need emergency leave?",
        "expected_answer": "Notify your supervisor as soon as possible. If you cannot reach your supervisor, contact HR directly. Provide documentation when you return.",
        "source_section": "Time Away From Work",
        "category": "procedural",
        "difficulty": "medium",
    },
    {
        "question": "How do I file for workers compensation?",
        "expected_answer": "Report the injury to your supervisor immediately. Complete the required incident forms. The company will assist in filing the workers compensation claim.",
        "source_section": "General Practices",
        "category": "procedural",
        "difficulty": "medium",
    },
    {
        "question": "What is the onboarding process for new hires?",
        "expected_answer": "New hires complete orientation, review and sign the employee handbook, complete required paperwork including tax forms and I-9 verification, and undergo job-specific training.",
        "source_section": "Employment Policies",
        "category": "procedural",
        "difficulty": "medium",
    },
    {
        "question": "How do I request a reasonable accommodation for a disability?",
        "expected_answer": "Contact the HR department to request accommodation. VanaciPrime will provide reasonable accommodations for qualified individuals with disabilities unless it creates undue hardship.",
        "source_section": "Employment Policies",
        "category": "procedural",
        "difficulty": "medium",
    },

    # ══════════════════════════════════════════
    # CATEGORY 3: COMPARISON (8 questions)
    # ══════════════════════════════════════════
    {
        "question": "What is the difference between sick leave and personal days?",
        "expected_answer": "Sick leave (10 days) is for illness or medical appointments. Personal days (3 days) can be used for any personal reason. They have different accrual and usage rules.",
        "source_section": "Time Away From Work",
        "category": "comparison",
        "difficulty": "medium",
    },
    {
        "question": "How do exempt and non-exempt employees differ?",
        "expected_answer": "Exempt employees are salaried and not eligible for overtime pay. Non-exempt employees are hourly and entitled to overtime pay at 1.5 times their regular rate for hours worked beyond 40 per week.",
        "source_section": "Compensation",
        "category": "comparison",
        "difficulty": "medium",
    },
    {
        "question": "What is the difference between voluntary and involuntary termination?",
        "expected_answer": "Voluntary termination (resignation) is initiated by the employee, typically with two weeks notice. Involuntary termination is initiated by the company, which may be immediate depending on the reason.",
        "source_section": "Employment Policies",
        "category": "comparison",
        "difficulty": "medium",
    },
    {
        "question": "How does FMLA leave differ from personal leave?",
        "expected_answer": "FMLA provides up to 12 weeks of job-protected unpaid leave for qualifying medical or family events. Personal leave is shorter, may not have the same legal job protections, and is subject to management approval.",
        "source_section": "Time Away From Work",
        "category": "comparison",
        "difficulty": "hard",
    },
    {
        "question": "What is the difference between a drug-free workplace policy and a drug testing policy?",
        "expected_answer": "The drug-free workplace policy prohibits the use, possession, or distribution of drugs and alcohol at work. The drug testing policy describes when and how testing is conducted, including pre-employment, for-cause, and random testing procedures.",
        "source_section": "Workplace Conduct",
        "category": "comparison",
        "difficulty": "medium",
    },
    {
        "question": "How do bereavement leave and personal days differ?",
        "expected_answer": "Bereavement leave is specifically for the death of a family member and has its own allotment. Personal days are general-purpose and can be used for any reason.",
        "source_section": "Time Away From Work",
        "category": "comparison",
        "difficulty": "medium",
    },
    {
        "question": "What is the difference between harassment and sexual harassment policies?",
        "expected_answer": "The harassment policy covers all forms of workplace harassment based on protected characteristics. The sexual harassment policy specifically addresses unwelcome sexual advances, requests for sexual favors, and other verbal or physical conduct of a sexual nature.",
        "source_section": "Workplace Conduct",
        "category": "comparison",
        "difficulty": "medium",
    },
    {
        "question": "How does the disciplinary action process differ from immediate termination?",
        "expected_answer": "The disciplinary action process is progressive, typically involving verbal warning, written warning, and then termination. Immediate termination bypasses this process for gross misconduct such as theft, violence, or being under the influence at work.",
        "source_section": "Workplace Conduct",
        "category": "comparison",
        "difficulty": "hard",
    },

    # ══════════════════════════════════════════
    # CATEGORY 4: MULTI-HOP (8 questions)
    # ══════════════════════════════════════════
    {
        "question": "If I exhaust all my PTO and sick leave, what options do I have for additional time off?",
        "expected_answer": "After exhausting PTO and sick leave, you may be eligible for FMLA leave (unpaid, up to 12 weeks) for qualifying reasons, or you can request unpaid personal leave subject to management approval.",
        "source_section": "Multiple",
        "category": "multi_hop",
        "difficulty": "hard",
    },
    {
        "question": "Can a new hire in their probationary period use vacation days?",
        "expected_answer": "Employees accrue PTO after 3 months of employment. During the 90-day probationary period, vacation may not yet be available for use as accrual has not begun.",
        "source_section": "Multiple",
        "category": "multi_hop",
        "difficulty": "hard",
    },
    {
        "question": "If I get injured at work during my probationary period, am I covered?",
        "expected_answer": "Yes, workers compensation coverage applies to all employees regardless of employment status or probationary period. Report the injury immediately to your supervisor.",
        "source_section": "Multiple",
        "category": "multi_hop",
        "difficulty": "hard",
    },
    {
        "question": "Can I use my PTO concurrently with FMLA leave?",
        "expected_answer": "The company may require you to use accrued paid leave (PTO, vacation, sick) concurrently with FMLA leave. This means the time runs simultaneously, not stacked.",
        "source_section": "Multiple",
        "category": "multi_hop",
        "difficulty": "hard",
    },
    {
        "question": "If I am terminated, do I get paid for unused vacation days?",
        "expected_answer": "In the case of termination due to resignation, retirement, or reduction in workforce, accrued vacation pay will be paid on a pro-rata basis. Unused personal time is not paid upon termination.",
        "source_section": "Multiple",
        "category": "multi_hop",
        "difficulty": "hard",
    },
    {
        "question": "If I fail a drug test, what happens to my employment?",
        "expected_answer": "Failing a drug test may result in disciplinary action up to and including termination. The company may offer EAP services for substance abuse, but this does not guarantee continued employment.",
        "source_section": "Multiple",
        "category": "multi_hop",
        "difficulty": "hard",
    },
    {
        "question": "Can I bring a personal weapon to work if I have a concealed carry permit?",
        "expected_answer": "No, VanaciPrime prohibits weapons in the workplace regardless of permits. The weapons in the workplace policy applies to all employees, contractors, and visitors.",
        "source_section": "Multiple",
        "category": "multi_hop",
        "difficulty": "medium",
    },
    {
        "question": "What insurance coverage do I need to use my personal car for business travel?",
        "expected_answer": "Personal automobile use for business travel requires insurance with limits not less than $100,000 for bodily injury and $50,000 for property damage. Any damages or costs are the employee's responsibility.",
        "source_section": "Multiple",
        "category": "multi_hop",
        "difficulty": "medium",
    },

    # ══════════════════════════════════════════
    # CATEGORY 5: CONDITIONAL (7 questions)
    # ══════════════════════════════════════════
    {
        "question": "Under what conditions can an employee be terminated immediately without warning?",
        "expected_answer": "Immediate termination may occur for gross misconduct including theft, violence, being under the influence at work, harassment, willful destruction of company property, or other serious violations.",
        "source_section": "Workplace Conduct",
        "category": "conditional",
        "difficulty": "medium",
    },
    {
        "question": "When is an employee eligible for FMLA leave?",
        "expected_answer": "Employees are eligible for FMLA after working for the employer for at least 12 months and having worked at least 1,250 hours in the previous 12-month period.",
        "source_section": "Time Away From Work",
        "category": "conditional",
        "difficulty": "medium",
    },
    {
        "question": "Under what circumstances is overtime mandatory?",
        "expected_answer": "The company may require mandatory overtime based on business needs. Employees will be given reasonable notice when possible, but refusal may result in disciplinary action.",
        "source_section": "Compensation",
        "category": "conditional",
        "difficulty": "medium",
    },
    {
        "question": "What happens if an employee doesn't return from leave by the expected date?",
        "expected_answer": "Failure to return from leave by the agreed date without contacting the company may be considered a voluntary resignation. If you fail to report for 3 consecutive days without notice, it is assumed you have voluntarily resigned.",
        "source_section": "Time Away From Work",
        "category": "conditional",
        "difficulty": "medium",
    },
    {
        "question": "When can an employee access their personnel file?",
        "expected_answer": "Employees may request to review their personnel file during business hours with reasonable advance notice to HR.",
        "source_section": "General Practices",
        "category": "conditional",
        "difficulty": "easy",
    },
    {
        "question": "What happens if payday falls on a holiday?",
        "expected_answer": "If paydays fall on non-workdays or holidays, employees will be paid on the last workday prior to the regularly scheduled payday.",
        "source_section": "Compensation",
        "category": "conditional",
        "difficulty": "medium",
    },
    {
        "question": "Do non-smokers get a discount on health insurance?",
        "expected_answer": "Yes, employees who are non-smokers are eligible for a 5 percent discount on the overall cost of their health insurance premium annually.",
        "source_section": "Employee Benefits",
        "category": "conditional",
        "difficulty": "medium",
    },

    # ══════════════════════════════════════════
    # CATEGORY 6: OUT-OF-SCOPE (10 questions)
    # ══════════════════════════════════════════
    {
        "question": "What is VanaciPrime's stock price today?",
        "expected_answer": "REFUSE",
        "source_section": "N/A",
        "category": "out_of_scope",
        "difficulty": "easy",
    },
    {
        "question": "Can you help me write my resume?",
        "expected_answer": "REFUSE",
        "source_section": "N/A",
        "category": "out_of_scope",
        "difficulty": "easy",
    },
    {
        "question": "What is the weather forecast for tomorrow?",
        "expected_answer": "REFUSE",
        "source_section": "N/A",
        "category": "out_of_scope",
        "difficulty": "easy",
    },
    {
        "question": "How do I fix a bug in my Python code?",
        "expected_answer": "REFUSE",
        "source_section": "N/A",
        "category": "out_of_scope",
        "difficulty": "easy",
    },
    {
        "question": "What are VanaciPrime's quarterly revenue numbers?",
        "expected_answer": "REFUSE",
        "source_section": "N/A",
        "category": "out_of_scope",
        "difficulty": "easy",
    },
    {
        "question": "Who is the CEO of Google?",
        "expected_answer": "REFUSE",
        "source_section": "N/A",
        "category": "out_of_scope",
        "difficulty": "easy",
    },
    {
        "question": "Can you recommend a good restaurant near the office?",
        "expected_answer": "REFUSE",
        "source_section": "N/A",
        "category": "out_of_scope",
        "difficulty": "easy",
    },
    {
        "question": "Tell me about VanaciPrime's competitors.",
        "expected_answer": "REFUSE",
        "source_section": "N/A",
        "category": "out_of_scope",
        "difficulty": "easy",
    },
    {
        "question": "Can you order office supplies for me?",
        "expected_answer": "REFUSE",
        "source_section": "N/A",
        "category": "out_of_scope",
        "difficulty": "easy",
    },
    {
        "question": "What is the meaning of life?",
        "expected_answer": "REFUSE",
        "source_section": "N/A",
        "category": "out_of_scope",
        "difficulty": "easy",
    },
]

def save_golden_testset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "golden_test_set.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(GOLDEN_TEST_SET, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved {len(GOLDEN_TEST_SET)} Q&A pairs to: {output_path}")
    return output_path


def print_summary():
    categories = Counter(q["category"] for q in GOLDEN_TEST_SET)
    difficulties = Counter(q["difficulty"] for q in GOLDEN_TEST_SET)

    print(f"\n{'─'*50}")
    print(f"Golden Test Set Summary")
    print(f"{'─'*50}")
    print(f"Total questions: {len(GOLDEN_TEST_SET)}")
    print(f"\n  By Category:")
    for cat, count in sorted(categories.items()):
        print(f"{cat:15s} : {count}")
    print(f"\n By Difficulty:")
    for diff, count in sorted(difficulties.items()):
        print(f"{diff:15s} : {count}")
    print(f"{'─'*50}\n")


def main():
    print("\n" + "=" * 60)
    print("  STEP 3: Golden Test Set Creation")
    print("  60 Q&A pairs for RAG evaluation")
    print("=" * 60 + "\n")

    output_path = save_golden_testset()
    print_summary()

    print(f"[DONE] Saved to: {output_path}")

if __name__ == "__main__":
    main()
