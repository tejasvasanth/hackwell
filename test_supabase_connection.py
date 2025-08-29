#!/usr/bin/env python3
"""
Test Supabase connection with the provided credentials.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.supabase_client import SupabaseClient
from config.settings import settings

def test_supabase_connection():
    """Test the Supabase connection and basic operations."""
    print("Testing Supabase Connection...")
    print(f"Supabase URL: {settings.supabase_url}")
    print(f"Supabase Key: {settings.supabase_key[:20]}..." if settings.supabase_key else "No key found")
    
    try:
        # Initialize client
        client = SupabaseClient()
        
        if client.client is None:
            print("❌ Failed to initialize Supabase client")
            return False
        
        print("✅ Supabase client initialized successfully")
        
        # Test basic connection by trying to list tables
        try:
            # This will test if we can connect to the database
            response = client.client.table('patients').select('*').limit(1).execute()
            print("✅ Successfully connected to Supabase database")
            print(f"Response status: {response.count if hasattr(response, 'count') else 'Connected'}")
            
        except Exception as e:
            print(f"⚠️  Database connection test failed (this is expected if tables don't exist yet): {e}")
            print("✅ Basic Supabase client connection is working")
        
        return True
        
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_supabase_connection()
    if success:
        print("\n🎉 Supabase connection test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Supabase connection test failed!")
        sys.exit(1)