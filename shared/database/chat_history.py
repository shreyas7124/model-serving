"""
Shared chat history module for all applications
Stores conversation history per user session
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

class ChatHistory:
    def __init__(self, db_path: str = "shared/database/chat_history.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the chat history database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                app_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_conversation(self, user_id: Optional[int], app_name: str) -> int:
        """Create a new conversation and return its ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO conversations (user_id, app_name) VALUES (?, ?)',
            (user_id, app_name)
        )
        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return conversation_id
    
    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to a conversation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute(
            'INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, ?, ?, ?)',
            (conversation_id, role, content, metadata_json)
        )
        
        # Update conversation timestamp
        cursor.execute(
            'UPDATE conversations SET updated_at = ? WHERE id = ?',
            (datetime.now(), conversation_id)
        )
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, conversation_id: int) -> List[Dict[str, Any]]:
        """Get all messages from a conversation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT role, content, metadata, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at',
            (conversation_id,)
        )
        
        messages = []
        for row in cursor.fetchall():
            message = {
                'role': row[0],
                'content': row[1],
                'metadata': json.loads(row[2]) if row[2] else None,
                'created_at': row[3]
            }
            messages.append(message)
        
        conn.close()
        return messages
    
    def get_user_conversations(self, user_id: int, app_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all conversations for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if app_name:
            cursor.execute(
                'SELECT id, app_name, created_at, updated_at FROM conversations WHERE user_id = ? AND app_name = ? ORDER BY updated_at DESC',
                (user_id, app_name)
            )
        else:
            cursor.execute(
                'SELECT id, app_name, created_at, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC',
                (user_id,)
            )
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                'id': row[0],
                'app_name': row[1],
                'created_at': row[2],
                'updated_at': row[3]
            })
        
        conn.close()
        return conversations
    
    def delete_conversation(self, conversation_id: int):
        """Delete a conversation and all its messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        
        conn.commit()
        conn.close()
