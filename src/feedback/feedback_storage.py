# src/feedback/feedback_storage.py

"""
Storage backends for the Feedback Loop system.

This module provides different storage options for feedback data including
JSON files and SQLite database.
"""

import json
import sqlite3
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import threading
from collections import defaultdict

from .feedback_models import (
    FeedbackEvent, AgentCorrection, CustomerFeedback,
    FeedbackType, FeedbackSeverity, PredictionQuality
)

logger = logging.getLogger(__name__)


class FeedbackStorage(ABC):
    """Abstract base class for feedback storage backends."""
    
    @abstractmethod
    def store_correction(self, correction: AgentCorrection) -> bool:
        """Store an agent correction."""
        pass
    
    @abstractmethod
    def store_customer_feedback(self, feedback: CustomerFeedback) -> bool:
        """Store customer feedback."""
        pass
    
    @abstractmethod
    def store_event(self, event: FeedbackEvent) -> bool:
        """Store a feedback event."""
        pass
    
    @abstractmethod
    def get_corrections(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ticket_id: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> List[AgentCorrection]:
        """Retrieve agent corrections with filters."""
        pass
    
    @abstractmethod
    def get_customer_feedback(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ticket_id: Optional[str] = None,
        rating_min: Optional[int] = None
    ) -> List[CustomerFeedback]:
        """Retrieve customer feedback with filters."""
        pass
    
    @abstractmethod
    def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[FeedbackType] = None
    ) -> List[FeedbackEvent]:
        """Retrieve feedback events with filters."""
        pass
    
    @abstractmethod
    def get_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get aggregate statistics."""
        pass


class JSONFeedbackStorage(FeedbackStorage):
    """JSON file-based storage for feedback data."""
    
    def __init__(self, storage_dir: str = "feedback_data"):
        """Initialize JSON storage."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.corrections_file = self.storage_dir / "corrections.jsonl"
        self.feedback_file = self.storage_dir / "customer_feedback.jsonl"
        self.events_file = self.storage_dir / "events.jsonl"
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"JSON feedback storage initialized: {self.storage_dir}")
    
    def store_correction(self, correction: AgentCorrection) -> bool:
        """Store an agent correction to JSON file."""
        try:
            with self._lock:
                with open(self.corrections_file, 'a', encoding='utf-8') as f:
                    json.dump(correction.to_dict(), f, ensure_ascii=False)
                    f.write('\n')
            
            logger.debug(f"Stored correction: {correction.correction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store correction: {e}")
            return False
    
    def store_customer_feedback(self, feedback: CustomerFeedback) -> bool:
        """Store customer feedback to JSON file."""
        try:
            with self._lock:
                with open(self.feedback_file, 'a', encoding='utf-8') as f:
                    json.dump(feedback.to_dict(), f, ensure_ascii=False)
                    f.write('\n')
            
            logger.debug(f"Stored customer feedback: {feedback.feedback_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store customer feedback: {e}")
            return False
    
    def store_event(self, event: FeedbackEvent) -> bool:
        """Store a feedback event to JSON file."""
        try:
            with self._lock:
                with open(self.events_file, 'a', encoding='utf-8') as f:
                    json.dump(event.to_dict(), f, ensure_ascii=False)
                    f.write('\n')
            
            logger.debug(f"Stored event: {event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return False
    
    def get_corrections(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ticket_id: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> List[AgentCorrection]:
        """Retrieve agent corrections with filters."""
        corrections = []
        
        try:
            if not self.corrections_file.exists():
                return corrections
            
            with open(self.corrections_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        correction = AgentCorrection.from_dict(data)
                        
                        # Apply filters
                        if start_date and correction.correction_timestamp < start_date:
                            continue
                        if end_date and correction.correction_timestamp > end_date:
                            continue
                        if ticket_id and correction.ticket_id != ticket_id:
                            continue
                        if model_type and correction.model_type != model_type:
                            continue
                        
                        corrections.append(correction)
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse correction line: {e}")
                        continue
            
            logger.debug(f"Retrieved {len(corrections)} corrections")
            return corrections
            
        except Exception as e:
            logger.error(f"Failed to retrieve corrections: {e}")
            return []
    
    def get_customer_feedback(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ticket_id: Optional[str] = None,
        rating_min: Optional[int] = None
    ) -> List[CustomerFeedback]:
        """Retrieve customer feedback with filters."""
        feedback_list = []
        
        try:
            if not self.feedback_file.exists():
                return feedback_list
            
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        feedback = CustomerFeedback.from_dict(data)
                        
                        # Apply filters
                        if start_date and feedback.feedback_timestamp < start_date:
                            continue
                        if end_date and feedback.feedback_timestamp > end_date:
                            continue
                        if ticket_id and feedback.ticket_id != ticket_id:
                            continue
                        if rating_min and feedback.rating < rating_min:
                            continue
                        
                        feedback_list.append(feedback)
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse feedback line: {e}")
                        continue
            
            logger.debug(f"Retrieved {len(feedback_list)} feedback entries")
            return feedback_list
            
        except Exception as e:
            logger.error(f"Failed to retrieve customer feedback: {e}")
            return []
    
    def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[FeedbackType] = None
    ) -> List[FeedbackEvent]:
        """Retrieve feedback events with filters."""
        events = []
        
        try:
            if not self.events_file.exists():
                return events
            
            with open(self.events_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        event = FeedbackEvent.from_dict(data)
                        
                        # Apply filters
                        if start_date and event.timestamp < start_date:
                            continue
                        if end_date and event.timestamp > end_date:
                            continue
                        if event_type and event.event_type != event_type:
                            continue
                        
                        events.append(event)
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse event line: {e}")
                        continue
            
            logger.debug(f"Retrieved {len(events)} events")
            return events
            
        except Exception as e:
            logger.error(f"Failed to retrieve events: {e}")
            return []
    
    def get_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get aggregate statistics."""
        try:
            corrections = self.get_corrections(start_date, end_date)
            feedback_list = self.get_customer_feedback(start_date, end_date)
            
            stats = {
                "corrections": {
                    "total": len(corrections),
                    "by_model": defaultdict(int),
                    "by_category": defaultdict(int),
                    "by_severity": defaultdict(int),
                    "avg_confidence": 0.0
                },
                "customer_feedback": {
                    "total": len(feedback_list),
                    "avg_rating": 0.0,
                    "by_rating": defaultdict(int),
                    "satisfaction_breakdown": defaultdict(int),
                    "ai_usage_rate": 0.0
                }
            }
            
            # Analyze corrections
            if corrections:
                total_confidence = 0.0
                for correction in corrections:
                    stats["corrections"]["by_model"][correction.model_type] += 1
                    stats["corrections"]["by_category"][correction.original_prediction] += 1
                    stats["corrections"]["by_severity"][correction.severity.value] += 1
                    total_confidence += correction.original_confidence
                
                stats["corrections"]["avg_confidence"] = total_confidence / len(corrections)
            
            # Analyze customer feedback
            if feedback_list:
                total_rating = 0.0
                ai_used_count = 0
                
                for feedback in feedback_list:
                    stats["customer_feedback"]["by_rating"][feedback.rating] += 1
                    stats["customer_feedback"]["satisfaction_breakdown"][feedback.satisfaction_level] += 1
                    total_rating += feedback.rating
                    
                    if feedback.ai_suggestions_used:
                        ai_used_count += 1
                
                stats["customer_feedback"]["avg_rating"] = total_rating / len(feedback_list)
                stats["customer_feedback"]["ai_usage_rate"] = ai_used_count / len(feedback_list)
            
            # Convert defaultdicts to regular dicts
            for category in ["corrections", "customer_feedback"]:
                for key, value in stats[category].items():
                    if isinstance(value, defaultdict):
                        stats[category][key] = dict(value)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate stats: {e}")
            return {}


class SQLiteFeedbackStorage(FeedbackStorage):
    """SQLite database storage for feedback data."""
    
    def __init__(self, db_path: str = "feedback_data/feedback.db"):
        """Initialize SQLite storage."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        logger.info(f"SQLite feedback storage initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize database tables."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Agent corrections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_corrections (
                    correction_id TEXT PRIMARY KEY,
                    ticket_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    original_prediction TEXT NOT NULL,
                    original_confidence REAL NOT NULL,
                    model_type TEXT NOT NULL,
                    corrected_label TEXT NOT NULL,
                    correction_reason TEXT NOT NULL,
                    correction_notes TEXT,
                    prediction_quality TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    correction_timestamp TEXT NOT NULL,
                    ticket_data TEXT,
                    should_retrain BOOLEAN DEFAULT 1,
                    correction_confidence REAL DEFAULT 1.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Customer feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customer_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    ticket_id TEXT NOT NULL,
                    customer_id TEXT,
                    rating INTEGER NOT NULL,
                    comments TEXT,
                    feedback_type TEXT DEFAULT 'resolution_quality',
                    resolution_helpful BOOLEAN DEFAULT 1,
                    resolution_accurate BOOLEAN DEFAULT 1,
                    would_recommend BOOLEAN,
                    ai_suggestions_used BOOLEAN DEFAULT 0,
                    ai_suggestions_helpful BOOLEAN,
                    ai_accuracy_rating INTEGER,
                    feedback_timestamp TEXT NOT NULL,
                    resolution_time_hours REAL,
                    agent_id TEXT,
                    satisfaction_level TEXT NOT NULL,
                    needs_followup BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Feedback events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    ticket_id TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    processed BOOLEAN DEFAULT 0,
                    processing_notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_corrections_ticket ON agent_corrections(ticket_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_corrections_timestamp ON agent_corrections(correction_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_corrections_model ON agent_corrections(model_type)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_ticket ON customer_feedback(ticket_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON customer_feedback(feedback_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON customer_feedback(rating)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON feedback_events(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON feedback_events(timestamp)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def store_correction(self, correction: AgentCorrection) -> bool:
        """Store an agent correction to SQLite database."""
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO agent_corrections
                    (correction_id, ticket_id, agent_id, original_prediction, original_confidence,
                     model_type, corrected_label, correction_reason, correction_notes,
                     prediction_quality, severity, correction_timestamp, ticket_data,
                     should_retrain, correction_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    correction.correction_id,
                    correction.ticket_id,
                    correction.agent_id,
                    correction.original_prediction,
                    correction.original_confidence,
                    correction.model_type,
                    correction.corrected_label,
                    correction.correction_reason,
                    correction.correction_notes,
                    correction.prediction_quality.value,
                    correction.severity.value,
                    correction.correction_timestamp.isoformat(),
                    json.dumps(correction.ticket_data) if correction.ticket_data else None,
                    correction.should_retrain,
                    correction.correction_confidence
                ))
                
                conn.commit()
                conn.close()
            
            logger.debug(f"Stored correction in database: {correction.correction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store correction in database: {e}")
            return False
    
    def store_customer_feedback(self, feedback: CustomerFeedback) -> bool:
        """Store customer feedback to SQLite database."""
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO customer_feedback
                    (feedback_id, ticket_id, customer_id, rating, comments, feedback_type,
                     resolution_helpful, resolution_accurate, would_recommend,
                     ai_suggestions_used, ai_suggestions_helpful, ai_accuracy_rating,
                     feedback_timestamp, resolution_time_hours, agent_id,
                     satisfaction_level, needs_followup)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.feedback_id,
                    feedback.ticket_id,
                    feedback.customer_id,
                    feedback.rating,
                    feedback.comments,
                    feedback.feedback_type,
                    feedback.resolution_helpful,
                    feedback.resolution_accurate,
                    feedback.would_recommend,
                    feedback.ai_suggestions_used,
                    feedback.ai_suggestions_helpful,
                    feedback.ai_accuracy_rating,
                    feedback.feedback_timestamp.isoformat(),
                    feedback.resolution_time_hours,
                    feedback.agent_id,
                    feedback.satisfaction_level,
                    feedback.needs_followup
                ))
                
                conn.commit()
                conn.close()
            
            logger.debug(f"Stored customer feedback in database: {feedback.feedback_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store customer feedback in database: {e}")
            return False
    
    def store_event(self, event: FeedbackEvent) -> bool:
        """Store a feedback event to SQLite database."""
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO feedback_events
                    (event_id, event_type, ticket_id, event_data, severity,
                     timestamp, processed, processing_notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.event_type.value,
                    event.ticket_id,
                    json.dumps(event.event_data),
                    event.severity.value,
                    event.timestamp.isoformat(),
                    event.processed,
                    event.processing_notes
                ))
                
                conn.commit()
                conn.close()
            
            logger.debug(f"Stored event in database: {event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store event in database: {e}")
            return False
    
    def get_corrections(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ticket_id: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> List[AgentCorrection]:
        """Retrieve agent corrections with filters."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            query = "SELECT * FROM agent_corrections WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND correction_timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND correction_timestamp <= ?"
                params.append(end_date.isoformat())
            
            if ticket_id:
                query += " AND ticket_id = ?"
                params.append(ticket_id)
            
            if model_type:
                query += " AND model_type = ?"
                params.append(model_type)
            
            query += " ORDER BY correction_timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            corrections = []
            for row in rows:
                # Map database row to AgentCorrection object
                correction_data = {
                    "correction_id": row[0],
                    "ticket_id": row[1],
                    "agent_id": row[2],
                    "original_prediction": row[3],
                    "original_confidence": row[4],
                    "model_type": row[5],
                    "corrected_label": row[6],
                    "correction_reason": row[7],
                    "correction_notes": row[8],
                    "prediction_quality": PredictionQuality(row[9]),
                    "severity": FeedbackSeverity(row[10]),
                    "correction_timestamp": datetime.fromisoformat(row[11]),
                    "ticket_data": json.loads(row[12]) if row[12] else None,
                    "should_retrain": bool(row[13]),
                    "correction_confidence": row[14]
                }
                
                corrections.append(AgentCorrection(**correction_data))
            
            logger.debug(f"Retrieved {len(corrections)} corrections from database")
            return corrections
            
        except Exception as e:
            logger.error(f"Failed to retrieve corrections from database: {e}")
            return []
    
    def get_customer_feedback(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ticket_id: Optional[str] = None,
        rating_min: Optional[int] = None
    ) -> List[CustomerFeedback]:
        """Retrieve customer feedback with filters."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            query = "SELECT * FROM customer_feedback WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND feedback_timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND feedback_timestamp <= ?"
                params.append(end_date.isoformat())
            
            if ticket_id:
                query += " AND ticket_id = ?"
                params.append(ticket_id)
            
            if rating_min:
                query += " AND rating >= ?"
                params.append(rating_min)
            
            query += " ORDER BY feedback_timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            feedback_list = []
            for row in rows:
                # Map database row to CustomerFeedback object
                feedback_data = {
                    "feedback_id": row[0],
                    "ticket_id": row[1],
                    "customer_id": row[2],
                    "rating": row[3],
                    "comments": row[4],
                    "feedback_type": row[5],
                    "resolution_helpful": bool(row[6]),
                    "resolution_accurate": bool(row[7]),
                    "would_recommend": bool(row[8]) if row[8] is not None else None,
                    "ai_suggestions_used": bool(row[9]),
                    "ai_suggestions_helpful": bool(row[10]) if row[10] is not None else None,
                    "ai_accuracy_rating": row[11],
                    "feedback_timestamp": datetime.fromisoformat(row[12]),
                    "resolution_time_hours": row[13],
                    "agent_id": row[14]
                }
                
                feedback_list.append(CustomerFeedback(**feedback_data))
            
            logger.debug(f"Retrieved {len(feedback_list)} feedback entries from database")
            return feedback_list
            
        except Exception as e:
            logger.error(f"Failed to retrieve customer feedback from database: {e}")
            return []
    
    def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[FeedbackType] = None
    ) -> List[FeedbackEvent]:
        """Retrieve feedback events with filters."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            query = "SELECT * FROM feedback_events WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type.value)
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            events = []
            for row in rows:
                # Map database row to FeedbackEvent object
                event_data = {
                    "event_id": row[0],
                    "event_type": FeedbackType(row[1]),
                    "ticket_id": row[2],
                    "event_data": json.loads(row[3]),
                    "severity": FeedbackSeverity(row[4]),
                    "timestamp": datetime.fromisoformat(row[5]),
                    "processed": bool(row[6]),
                    "processing_notes": row[7]
                }
                
                events.append(FeedbackEvent(**event_data))
            
            logger.debug(f"Retrieved {len(events)} events from database")
            return events
            
        except Exception as e:
            logger.error(f"Failed to retrieve events from database: {e}")
            return []
    
    def get_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get aggregate statistics from database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Build date filter
            date_filter = ""
            params = []
            
            if start_date:
                date_filter += " AND correction_timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                date_filter += " AND correction_timestamp <= ?"
                params.append(end_date.isoformat())
            
            # Correction stats
            cursor.execute(f"""
                SELECT COUNT(*), AVG(original_confidence), model_type, 
                       COUNT(*) as count
                FROM agent_corrections 
                WHERE 1=1 {date_filter}
                GROUP BY model_type
            """, params)
            
            correction_stats = cursor.fetchall()
            
            # Customer feedback stats
            feedback_params = []
            feedback_date_filter = ""
            
            if start_date:
                feedback_date_filter += " AND feedback_timestamp >= ?"
                feedback_params.append(start_date.isoformat())
            
            if end_date:
                feedback_date_filter += " AND feedback_timestamp <= ?"
                feedback_params.append(end_date.isoformat())
            
            cursor.execute(f"""
                SELECT COUNT(*), AVG(rating), satisfaction_level,
                       COUNT(*) as count
                FROM customer_feedback 
                WHERE 1=1 {feedback_date_filter}
                GROUP BY satisfaction_level
            """, feedback_params)
            
            feedback_stats = cursor.fetchall()
            
            conn.close()
            
            # Process results
            stats = {
                "corrections": {
                    "total": sum(row[0] for row in correction_stats),
                    "by_model": {row[2]: row[3] for row in correction_stats},
                    "avg_confidence": sum(row[1] * row[3] for row in correction_stats) / 
                                    sum(row[3] for row in correction_stats) if correction_stats else 0.0
                },
                "customer_feedback": {
                    "total": sum(row[0] for row in feedback_stats),
                    "by_satisfaction": {row[2]: row[3] for row in feedback_stats},
                    "avg_rating": sum(row[1] * row[3] for row in feedback_stats) / 
                                 sum(row[3] for row in feedback_stats) if feedback_stats else 0.0
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate stats from database: {e}")
            return {}