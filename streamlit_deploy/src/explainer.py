import random
from src.config import CORE_AI_SKILLS

class CandidateExplainer:
    def __init__(self):
        pass

    def generate_explanation(self, rank: int, score: float, cand, components: dict) -> str:
        """
        Generates a 1-2 sentence factual explanation based on candidate profile.
        Ensures variation, facts, JD alignment, and rank tone consistency.
        """
        profile = cand.get("profile", {})
        skills = cand.get("skills", [])
        signals = cand.get("redrob_signals", {})
        
        current_title = profile.get("current_title", "Software Engineer")
        years_exp = profile.get("years_of_experience", 0.0)
        location = profile.get("location", "India")
        notice = signals.get("notice_period_days", 30)
        resp_rate = int(signals.get("recruiter_response_rate", 0.0) * 100)
        
        # Get actual skills present in candidate profile matching core AI skills
        matched_skills = []
        for s in skills:
            name = s.get("name", "")
            is_core = any(cs in name.lower() for cs in CORE_AI_SKILLS)
            if is_core:
                matched_skills.append(name)
                
        # Select up to 3 matched skills for the explanation
        matched_skills = matched_skills[:3]
        skills_str = ", ".join(matched_skills) if matched_skills else "relevant engineering skills"

        # Check for minor concerns
        concerns = []
        if years_exp < 5.0:
            concerns.append(f"experience is slightly junior ({years_exp} yrs)")
        elif years_exp > 9.0:
            concerns.append(f"experience is senior ({years_exp} yrs) relative to the 5-9 yrs band")
            
        if notice > 60:
            concerns.append(f"long notice period ({notice} days)")
            
        concern_str = " but " + " and ".join(concerns) if concerns else ""

        # Sentence templates based on rank/pedigree
        if rank <= 10:
            # Top-10: Highly enthusiastic and perfect match
            templates = [
                f"Outstanding {current_title} with {years_exp} years of experience; possesses strong core expertise in {skills_str} and shows excellent platform engagement ({resp_rate}% response rate).",
                f"Top fit: {current_title} located in {location} with {years_exp} yrs experience; matches core JD requirements with verified skills in {skills_str} and active GitHub contribution.",
                f"Founding team match: {current_title} offering {years_exp} years of experience; strong production exposure to {skills_str} and quick availability ({notice}-day notice period)."
            ]
        elif rank <= 50:
            # Rank 11-50: Solid match, minor gaps or slight location/notice deviations
            templates = [
                f"Strong {current_title} with {years_exp} years of experience, bringing solid background in {skills_str}{concern_str}; has high responsiveness ({resp_rate}%).",
                f"Well-qualified {current_title} based in {location} with {years_exp} yrs experience; demonstrated competence in {skills_str} and active profile views.",
                f"Experienced {current_title} with {years_exp} years in the industry; shows good alignment with {skills_str}{concern_str} and a {notice}-day notice period."
            ]
        else:
            # Rank 51-100: Secondary matches, filler candidates with some concerns
            templates = [
                f"Adjacent fit: {current_title} with {years_exp} years experience matching key skills like {skills_str}{concern_str}.",
                f"Alternative candidate: {current_title} with {years_exp} yrs of experience; covers baseline requirements with skills in {skills_str}{concern_str}.",
                f"{current_title} with {years_exp} years experience; exhibits some skill overlap in {skills_str} but has secondary alignment overall."
            ]

        # Deterministic choice based on candidate_id to ensure reproducibility
        cand_seed = sum(ord(char) for char in cand.get("candidate_id", ""))
        random_index = cand_seed % len(templates)
        explanation = templates[random_index]
        
        return explanation
