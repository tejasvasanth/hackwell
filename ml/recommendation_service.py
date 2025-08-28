from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
import threading
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class RecommendationType(Enum):
    SCHEDULE_VISIT = "schedule_visit"
    LIFESTYLE_CHANGE = "lifestyle_change"
    ADJUST_MEDICATION = "adjust_medication"
    EMERGENCY_CARE = "emergency_care"
    MONITORING = "monitoring"
    PREVENTION = "prevention"

@dataclass
class Recommendation:
    type: RecommendationType
    priority: str  # "urgent", "high", "medium", "low"
    title: str
    description: str
    timeline: str  # "immediate", "within_24h", "within_week", "within_month"
    category: str  # "medical", "lifestyle", "medication", "monitoring"
    confidence: float  # 0.0 to 1.0
    reasoning: str

class RecommendationService:
    def __init__(self):
        self._lock = threading.Lock()
        self.risk_thresholds = {
            "critical": 0.9,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.0
        }
        logger.info("Recommendation service initialized")
    
    def get_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        if risk_score >= self.risk_thresholds["critical"]:
            return RiskLevel.CRITICAL
        elif risk_score >= self.risk_thresholds["high"]:
            return RiskLevel.HIGH
        elif risk_score >= self.risk_thresholds["medium"]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def generate_recommendations(
        self, 
        patient_id: str, 
        risk_score: float, 
        patient_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate personalized recommendations based on risk score and patient profile"""
        try:
            with self._lock:
                risk_level = self.get_risk_level(risk_score)
                recommendations = []
                
                # Extract patient profile data
                age = patient_profile.get("age", 50)
                cholesterol = patient_profile.get("cholesterol", 200)
                blood_pressure = patient_profile.get("blood_pressure", 120)
                diabetes = patient_profile.get("diabetes", 0)
                smoking = patient_profile.get("smoking", 0)
                family_history = patient_profile.get("family_history", 0)
                bmi = patient_profile.get("bmi", 25)
                exercise_frequency = patient_profile.get("exercise_frequency", 2)
                
                # Risk-based recommendations
                if risk_level == RiskLevel.CRITICAL:
                    recommendations.extend(self._get_critical_recommendations(patient_profile))
                elif risk_level == RiskLevel.HIGH:
                    recommendations.extend(self._get_high_risk_recommendations(patient_profile))
                elif risk_level == RiskLevel.MEDIUM:
                    recommendations.extend(self._get_medium_risk_recommendations(patient_profile))
                else:
                    recommendations.extend(self._get_low_risk_recommendations(patient_profile))
                
                # Factor-specific recommendations
                if cholesterol > 240:
                    recommendations.extend(self._get_cholesterol_recommendations(cholesterol))
                
                if blood_pressure > 140:
                    recommendations.extend(self._get_hypertension_recommendations(blood_pressure))
                
                if diabetes == 1:
                    recommendations.extend(self._get_diabetes_recommendations())
                
                if smoking == 1:
                    recommendations.extend(self._get_smoking_recommendations())
                
                if bmi > 30:
                    recommendations.extend(self._get_obesity_recommendations(bmi))
                
                if exercise_frequency < 2:
                    recommendations.extend(self._get_exercise_recommendations())
                
                # Sort by priority and confidence
                recommendations = self._prioritize_recommendations(recommendations)
                
                logger.info(f"Generated {len(recommendations)} recommendations for patient {patient_id}")
                return [rec.__dict__ if hasattr(rec, '__dict__') else rec for rec in recommendations]
                
        except Exception as e:
            logger.error(f"Error generating recommendations for patient {patient_id}: {str(e)}")
            return self._get_default_recommendations()
    
    def _get_critical_recommendations(self, profile: Dict[str, Any]) -> List[Recommendation]:
        """Recommendations for critical risk patients"""
        return [
            Recommendation(
                type=RecommendationType.EMERGENCY_CARE,
                priority="urgent",
                title="Immediate Medical Attention Required",
                description="Your cardiovascular risk assessment indicates critical risk. Please seek immediate medical evaluation.",
                timeline="immediate",
                category="medical",
                confidence=0.95,
                reasoning="Critical risk score requires immediate medical intervention"
            ),
            Recommendation(
                type=RecommendationType.SCHEDULE_VISIT,
                priority="urgent",
                title="Emergency Cardiology Consultation",
                description="Schedule an urgent appointment with a cardiologist within 24 hours.",
                timeline="within_24h",
                category="medical",
                confidence=0.9,
                reasoning="Critical cardiovascular risk requires specialist evaluation"
            )
        ]
    
    def _get_high_risk_recommendations(self, profile: Dict[str, Any]) -> List[Recommendation]:
        """Recommendations for high risk patients"""
        return [
            Recommendation(
                type=RecommendationType.SCHEDULE_VISIT,
                priority="high",
                title="Urgent Cardiology Consultation",
                description="Schedule an appointment with a cardiologist within one week for comprehensive evaluation.",
                timeline="within_week",
                category="medical",
                confidence=0.85,
                reasoning="High cardiovascular risk requires prompt medical attention"
            ),
            Recommendation(
                type=RecommendationType.ADJUST_MEDICATION,
                priority="high",
                title="Medication Review Required",
                description="Consult with your physician about starting or adjusting cardiovascular medications.",
                timeline="within_week",
                category="medication",
                confidence=0.8,
                reasoning="High risk may require pharmacological intervention"
            ),
            Recommendation(
                type=RecommendationType.MONITORING,
                priority="high",
                title="Intensive Health Monitoring",
                description="Begin daily monitoring of blood pressure, weight, and symptoms.",
                timeline="immediate",
                category="monitoring",
                confidence=0.75,
                reasoning="High risk patients benefit from close monitoring"
            )
        ]
    
    def _get_medium_risk_recommendations(self, profile: Dict[str, Any]) -> List[Recommendation]:
        """Recommendations for medium risk patients"""
        return [
            Recommendation(
                type=RecommendationType.SCHEDULE_VISIT,
                priority="medium",
                title="Preventive Cardiology Visit",
                description="Schedule a preventive cardiology consultation within one month.",
                timeline="within_month",
                category="medical",
                confidence=0.7,
                reasoning="Medium risk warrants preventive medical evaluation"
            ),
            Recommendation(
                type=RecommendationType.LIFESTYLE_CHANGE,
                priority="high",
                title="Comprehensive Lifestyle Modification",
                description="Implement heart-healthy diet, regular exercise, and stress management.",
                timeline="immediate",
                category="lifestyle",
                confidence=0.85,
                reasoning="Lifestyle changes are highly effective for medium risk patients"
            ),
            Recommendation(
                type=RecommendationType.MONITORING,
                priority="medium",
                title="Regular Health Monitoring",
                description="Monitor blood pressure and cholesterol levels monthly.",
                timeline="within_week",
                category="monitoring",
                confidence=0.7,
                reasoning="Regular monitoring helps track risk factor improvements"
            )
        ]
    
    def _get_low_risk_recommendations(self, profile: Dict[str, Any]) -> List[Recommendation]:
        """Recommendations for low risk patients"""
        return [
            Recommendation(
                type=RecommendationType.PREVENTION,
                priority="low",
                title="Maintain Heart-Healthy Lifestyle",
                description="Continue current healthy habits and maintain regular physical activity.",
                timeline="ongoing",
                category="lifestyle",
                confidence=0.8,
                reasoning="Prevention is key for maintaining low cardiovascular risk"
            ),
            Recommendation(
                type=RecommendationType.SCHEDULE_VISIT,
                priority="low",
                title="Annual Preventive Check-up",
                description="Schedule annual cardiovascular screening with your primary care physician.",
                timeline="within_year",
                category="medical",
                confidence=0.6,
                reasoning="Annual screening maintains preventive care"
            )
        ]
    
    def _get_cholesterol_recommendations(self, cholesterol: float) -> List[Recommendation]:
        """Cholesterol-specific recommendations"""
        severity = "severe" if cholesterol > 300 else "moderate"
        return [
            Recommendation(
                type=RecommendationType.LIFESTYLE_CHANGE,
                priority="high" if severity == "severe" else "medium",
                title="Cholesterol Management Program",
                description=f"Implement low-cholesterol diet and consider statin therapy. Current level: {cholesterol} mg/dL",
                timeline="immediate",
                category="lifestyle",
                confidence=0.9,
                reasoning=f"Cholesterol level of {cholesterol} mg/dL requires intervention"
            )
        ]
    
    def _get_hypertension_recommendations(self, bp: float) -> List[Recommendation]:
        """Blood pressure specific recommendations"""
        return [
            Recommendation(
                type=RecommendationType.LIFESTYLE_CHANGE,
                priority="high",
                title="Blood Pressure Management",
                description=f"Reduce sodium intake, increase physical activity. Current BP: {bp} mmHg",
                timeline="immediate",
                category="lifestyle",
                confidence=0.85,
                reasoning=f"Blood pressure of {bp} mmHg requires management"
            )
        ]
    
    def _get_diabetes_recommendations(self) -> List[Recommendation]:
        """Diabetes-specific recommendations"""
        return [
            Recommendation(
                type=RecommendationType.MONITORING,
                priority="high",
                title="Enhanced Cardiovascular Monitoring",
                description="Diabetes increases cardiovascular risk. Monitor HbA1c, blood pressure, and lipids closely.",
                timeline="ongoing",
                category="monitoring",
                confidence=0.9,
                reasoning="Diabetes significantly increases cardiovascular risk"
            )
        ]
    
    def _get_smoking_recommendations(self) -> List[Recommendation]:
        """Smoking cessation recommendations"""
        return [
            Recommendation(
                type=RecommendationType.LIFESTYLE_CHANGE,
                priority="urgent",
                title="Smoking Cessation Program",
                description="Enroll in a smoking cessation program immediately. Consider nicotine replacement therapy.",
                timeline="immediate",
                category="lifestyle",
                confidence=0.95,
                reasoning="Smoking is a major modifiable cardiovascular risk factor"
            )
        ]
    
    def _get_obesity_recommendations(self, bmi: float) -> List[Recommendation]:
        """Weight management recommendations"""
        return [
            Recommendation(
                type=RecommendationType.LIFESTYLE_CHANGE,
                priority="high",
                title="Weight Management Program",
                description=f"Target 5-10% weight loss through diet and exercise. Current BMI: {bmi}",
                timeline="within_month",
                category="lifestyle",
                confidence=0.8,
                reasoning=f"BMI of {bmi} increases cardiovascular risk"
            )
        ]
    
    def _get_exercise_recommendations(self) -> List[Recommendation]:
        """Exercise recommendations"""
        return [
            Recommendation(
                type=RecommendationType.LIFESTYLE_CHANGE,
                priority="medium",
                title="Increase Physical Activity",
                description="Aim for 150 minutes of moderate aerobic activity per week.",
                timeline="immediate",
                category="lifestyle",
                confidence=0.85,
                reasoning="Regular exercise significantly reduces cardiovascular risk"
            )
        ]
    
    def _prioritize_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Sort recommendations by priority and confidence"""
        priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        
        return sorted(
            recommendations,
            key=lambda x: (priority_order.get(x.priority, 4), -x.confidence)
        )
    
    def _get_default_recommendations(self) -> List[Dict[str, Any]]:
        """Default recommendations when error occurs"""
        return [
            {
                "type": "schedule_visit",
                "priority": "medium",
                "title": "General Health Check-up",
                "description": "Schedule a routine health check-up with your healthcare provider.",
                "timeline": "within_month",
                "category": "medical",
                "confidence": 0.5,
                "reasoning": "Default recommendation for health maintenance"
            }
        ]

# Global singleton instance
recommendation_service = RecommendationService()