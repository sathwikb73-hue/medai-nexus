-- MedAI Nexus — Initial DB Bootstrap
-- Runs once when PostgreSQL container first starts.
-- SQLAlchemy/Alembic handles the actual table creation via init_db().

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- for fast text search on reports

-- Confirm setup
SELECT 'MedAI Nexus database initialised at ' || NOW() AS status;
