import json
import gzip
import re
from datetime import datetime
from src.config import SERVICES_COMPANIES, NON_RELEVANT_TITLES, RELEVANT_TECH_KEYWORDS

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def is_honeypot(cand) -> bool:
    """
    Detects if a candidate record has structural inconsistencies,
    expert skill duration mismatch, or impossible salary parameters (trap profiles).
    """
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    # 1. Salary Min > Max Trap (physical impossibility)
    sal = signals.get("expected_salary_range_inr_lpa", {})
    if sal.get("min", 0) > sal.get("max", 0):
        return True
        
    # 2. Expert / Advanced Skills with 0 duration used (skill stuffing trap)
    expert_zero_dur_count = 0
    for s in skills:
        prof = s.get("proficiency", "").lower()
        dur = s.get("duration_months", 0)
        if prof in ["advanced", "expert"] and dur == 0:
            expert_zero_dur_count += 1
    if expert_zero_dur_count >= 5:
        return True
        
    # 3. Individual Job duration vs start/end calendar months mismatch
    ref_date = datetime(2026, 6, 29) # Current time reference
    for job in career:
        sd = parse_date(job.get("start_date"))
        ed = parse_date(job.get("end_date"))
        stated_dur = job.get("duration_months")
        if sd:
            end_ref = ed if ed else ref_date
            calendar_dur = (end_ref.year - sd.year) * 12 + (end_ref.month - sd.month)
            if stated_dur is not None and abs(stated_dur - calendar_dur) > 2:
                return True
                
    # 4. Job start date after end date (logical error)
    for job in career:
        sd = parse_date(job.get("start_date"))
        ed = parse_date(job.get("end_date"))
        if sd and ed and sd > ed:
            return True
            
    # 5. Stated experience vs calculated cumulative history duration mismatch
    stated_exp = profile.get("years_of_experience", 0)
    total_months = sum(j.get("duration_months", 0) for j in career)
    calc_years = total_months / 12.0
    if abs(stated_exp - calc_years) > 2.0:
        return True
        
    return False

def build_candidate_representation(cand) -> str:
    """
    Serializes a candidate profile into a compact text block for vector embeddings.
    Focuses on Title, Headline, Summary, and Skills to speed up CPU inference.
    """
    profile = cand.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    current_title = profile.get("current_title", "")
    current_industry = profile.get("current_industry", "")
    
    # Truncate long texts to speed up CPU tokenization and encoding
    summary_trunc = summary[:150] + "..." if len(summary) > 150 else summary
    headline_trunc = headline[:100] + "..." if len(headline) > 100 else headline
    
    # Skills list
    skills_list = [s.get("name", "") for s in cand.get("skills", [])]
    skills_str = ", ".join(skills_list[:15]) # limit to top 15 skills
    
    text = f"Title: {current_title}. Headline: {headline_trunc}. Summary: {summary_trunc}. Industry: {current_industry}. Skills: {skills_str}"
    
    # Clean spacing and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def iter_candidates(file_path):
    """
    Lazy loads candidates from candidates.jsonl or candidates.jsonl.gz.
    """
    path = str(file_path)
    if path.endswith(".gz"):
        f = gzip.open(path, "rt", encoding="utf-8")
    else:
        f = open(path, "r", encoding="utf-8")
        
    try:
        for line in f:
            if line.strip():
                yield json.loads(line)
    finally:
        f.close()
