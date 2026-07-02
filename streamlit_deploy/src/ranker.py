import numpy as np
from datetime import datetime
from src.config import (
    WEIGHT_SEMANTIC, WEIGHT_SKILLS, WEIGHT_EXPERIENCE, WEIGHT_BEHAVIOR, WEIGHT_EDUCATION,
    MIN_EXPERIENCE_YEARS, MAX_EXPERIENCE_YEARS, PREFERRED_LOCATIONS,
    SERVICES_COMPANIES, NON_RELEVANT_TITLES, RELEVANT_TECH_KEYWORDS, CORE_AI_SKILLS
)
from src.parser import is_honeypot, parse_date
from src.logger import setup_logger

logger = setup_logger()

class CandidateRanker:
    def __init__(self):
        pass

    def compute_experience_score(self, years_exp: float) -> float:
        """
        JD requires 5-9 years. Score falls off outside the target band.
        """
        if years_exp < 1.0:
            return 0.0  # Disqualify extremely junior profiles
        elif MIN_EXPERIENCE_YEARS <= years_exp <= MAX_EXPERIENCE_YEARS:
            return 1.0
        elif (MIN_EXPERIENCE_YEARS - 1.0) <= years_exp < MIN_EXPERIENCE_YEARS or MAX_EXPERIENCE_YEARS < years_exp <= (MAX_EXPERIENCE_YEARS + 2.0):
            return 0.7
        elif (MIN_EXPERIENCE_YEARS - 2.0) <= years_exp < (MIN_EXPERIENCE_YEARS - 1.0) or (MAX_EXPERIENCE_YEARS + 2.0) < years_exp <= (MAX_EXPERIENCE_YEARS + 4.0):
            return 0.4
        else:
            return 0.1

    def compute_skills_score(self, skills) -> float:
        """
        Computes skill alignment. Penalizes candidate if skill duration_months is 0
        (which indicates lazy keyword stuffing).
        """
        if not skills:
            return 0.0

        score_sum = 0.0
        matched_count = 0
        
        # Skill proficiency weights
        prof_weights = {
            "expert": 1.0,
            "advanced": 0.8,
            "intermediate": 0.6,
            "beginner": 0.3
        }

        for skill in skills:
            name = skill.get("name", "").lower()
            prof = skill.get("proficiency", "beginner").lower()
            dur = skill.get("duration_months", 0)
            
            # Check if it aligns with core AI skills
            is_core = any(cs in name for cs in CORE_AI_SKILLS)
            if is_core:
                matched_count += 1
                base_weight = prof_weights.get(prof, 0.3)
                
                # Trust factor: if duration is 0, give only 10% weight to that skill
                trust_mult = 1.0 if dur > 0 else 0.1
                score_sum += base_weight * trust_mult

        if matched_count == 0:
            return 0.0
            
        # Normalize and cap. Having 10 solid, well-aligned core skills gives a score of 1.0
        normalized_score = score_sum / 10.0
        return min(normalized_score, 1.0)

    def compute_education_score(self, education) -> float:
        """
        Ranks education pedigree based on tier and relevance of studies (CS/Data Science/Eng).
        """
        if not education:
            return 0.0

        max_edu_score = 0.0
        relevant_edu_keywords = ["computer", "data science", "ai", "artificial intelligence", "ml", "machine learning", "engineering", "mathematics", "statistics", "science"]

        for edu in education:
            degree = edu.get("degree", "").lower()
            field = edu.get("field_of_study", "").lower()
            tier = edu.get("tier", "unknown").lower()
            
            is_relevant = any(kw in field or kw in degree for kw in relevant_edu_keywords)
            
            # Base score by tier
            tier_scores = {
                "tier_1": 1.0 if is_relevant else 0.5,
                "tier_2": 0.8 if is_relevant else 0.3,
                "tier_3": 0.5 if is_relevant else 0.2,
                "tier_4": 0.2 if is_relevant else 0.1,
                "unknown": 0.2 if is_relevant else 0.1
            }
            
            score = tier_scores.get(tier, 0.1)
            if score > max_edu_score:
                max_edu_score = score
                
        return max_edu_score

    def compute_behavioral_score(self, signals) -> float:
        """
        Aggregates notice period, response rate, activity recency, and Github score.
        """
        if not signals:
            return 0.0

        # 1. Notice Period Score (prefer <= 30 days)
        notice = signals.get("notice_period_days", 180)
        if notice <= 30:
            notice_score = 1.0
        elif notice <= 60:
            notice_score = 0.7
        elif notice <= 90:
            notice_score = 0.4
        else:
            notice_score = 0.1

        # 2. Recruiter Response Rate
        response_rate = signals.get("recruiter_response_rate", 0.0)

        # 3. Last Active Date Recency (reference: 2026-06-29)
        ref_date = datetime(2026, 6, 29)
        last_active_str = signals.get("last_active_date")
        last_active = parse_date(last_active_str)
        if last_active:
            days_ago = (ref_date - last_active).days
            if days_ago <= 30:
                active_score = 1.0
            elif days_ago <= 90:
                active_score = 0.7
            elif days_ago <= 180:
                active_score = 0.4
            else:
                active_score = 0.1
        else:
            active_score = 0.1

        # 4. GitHub activity score
        gh = signals.get("github_activity_score", -1)
        gh_score = max(gh, 0) / 100.0 if gh >= 0 else 0.0

        # 5. Open to Work flag
        open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.5

        # Average the indicators
        behavioral_avg = (notice_score + response_rate + active_score + gh_score + open_to_work) / 5.0
        return behavioral_avg

    def has_only_services_history(self, career) -> bool:
        """
        Determines if the candidate has worked *exclusively* at services companies.
        """
        if not career:
            return False
            
        for job in career:
            company = job.get("company", "").lower()
            # If we find any company that is not a known services company, they have some product experience
            is_services = any(sc in company for sc in SERVICES_COMPANIES)
            if not is_services:
                return False
        return True

    def has_irrelevant_title(self, current_title: str) -> bool:
        """
        Checks if the current title is completely unrelated to AI/Tech (stuffer trap).
        """
        title_lower = current_title.lower()
        
        # If the title is explicitly in the non-relevant list, it's non-relevant
        is_non_relevant = any(nt in title_lower for nt in NON_RELEVANT_TITLES)
        
        # But if the title contains tech keywords, override (e.g. "Software Engineer" or "Data Scientist")
        is_tech = any(rk in title_lower for rk in RELEVANT_TECH_KEYWORDS)
        
        return is_non_relevant and not is_tech

    def score_candidate(self, cand, semantic_score: float) -> tuple:
        """
        Fuses all scores into a single final ranking score and provides the components details.
        """
        cid = cand["candidate_id"]
        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        skills = cand.get("skills", [])
        signals = cand.get("redrob_signals", {})
        
        # Core checks
        current_title = profile.get("current_title", "")
        years_exp = profile.get("years_of_experience", 0.0)
        location = profile.get("location", "").lower()
        country = profile.get("country", "").lower()
        relocate = signals.get("willing_to_relocate", False)

        # 1. Check for Honeypot
        if is_honeypot(cand):
            logger.debug(f"Candidate {cid} flagged as Honeypot. Disqualifying.")
            return 0.0, "honeypot", {}

        # 2. Compute individual components
        s_semantic = semantic_score
        s_skills = self.compute_skills_score(skills)
        s_exp = self.compute_experience_score(years_exp)
        s_behavior = self.compute_behavioral_score(signals)
        s_edu = self.compute_education_score(cand.get("education", []))

        # 3. Fuse scores (weights sum to 1.0)
        score = (
            WEIGHT_SEMANTIC * s_semantic +
            WEIGHT_SKILLS * s_skills +
            WEIGHT_EXPERIENCE * s_exp +
            WEIGHT_BEHAVIOR * s_behavior +
            WEIGHT_EDUCATION * s_edu
        )

        # 4. Multipliers for traps/filters
        multiplier = 1.0
        disqualifying_reason = None

        # A. Services Company check: if only worked at consulting/services
        if self.has_only_services_history(career):
            multiplier *= 0.05
            disqualifying_reason = "services_only"

        # B. Non-relevant Title check (e.g. HR Manager keyword stuffer)
        if self.has_irrelevant_title(current_title):
            multiplier *= 0.05
            disqualifying_reason = "irrelevant_title"

        score *= multiplier

        # 5. Location boost (Pune/Noida preferred, or within preferred cities)
        is_pref_location = any(loc in location or loc in country for loc in PREFERRED_LOCATIONS)
        if is_pref_location or relocate:
            score += 0.05
            
        score = min(max(score, 0.0), 1.0)

        components = {
            "semantic": s_semantic,
            "skills": s_skills,
            "experience": s_exp,
            "behavior": s_behavior,
            "education": s_edu,
            "disqualified": disqualifying_reason is not None,
            "reason": disqualifying_reason
        }

        return score, disqualifying_reason, components
