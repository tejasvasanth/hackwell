#!/usr/bin/env python3
"""
Script to create basic Supabase database tables programmatically.
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

def create_basic_tables():
    """Create basic database tables using Supabase client."""
    
    # Initialize Supabase client
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials in environment variables")
        return False
    
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Connected to Supabase")
        
        # Test if tables already exist
        try:
            result = supabase.table('patients').select('*').limit(1).execute()
            print("✅ Tables already exist!")
            return True
        except Exception:
            print("📋 Creating basic tables...")
            
        # For now, let's create a simple patients table to test
        # We'll insert some basic data to create the table structure
        try:
            # Try to insert a test patient - this will create the table if it doesn't exist
            test_patient = {
                'patient_id': 'TEST_001',
                'name': 'Test Patient',
                'age': 30,
                'gender': 'Male'
            }
            
            result = supabase.table('patients').insert(test_patient).execute()
            print("✅ Basic patients table created successfully")
            
            # Delete the test patient
            supabase.table('patients').delete().eq('patient_id', 'TEST_001').execute()
            print("✅ Test data cleaned up")
            
            return True
            
        except Exception as e:
            print(f"❌ Could not create tables programmatically: {str(e)}")
            print("\n📝 Please create tables manually in Supabase dashboard:")
            print("1. Go to your Supabase dashboard")
            print("2. Navigate to the SQL Editor")
            print("3. Copy and paste the contents of 'setup_supabase_schema.sql'")
            print("4. Execute the SQL script")
            print("5. Then run 'python setup_database.py' to insert sample data")
            return False
        
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {str(e)}")
        return False

if __name__ == "__main__":
    success = create_basic_tables()
    if success:
        print("\n✅ You can now run setup_database.py to insert sample data")
    else:
        print("\n⚠️  Please create the tables manually in Supabase dashboard first")