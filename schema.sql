-- Flood-Lead Intelligence MVP Database Schema
-- PostgreSQL with PostGIS extensions

-- Create required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Append-only raw detections table
CREATE TABLE IF NOT EXISTS flood_pixels_raw (
    id bigserial PRIMARY KEY,
    segment_id bigint NOT NULL,
    score smallint NOT NULL,
    homes integer NOT NULL,
    qpe_1h numeric NOT NULL,
    ffw boolean NOT NULL,
    geom geometry(Point, 4326) NOT NULL,
    first_seen timestamptz NOT NULL DEFAULT now()
);

-- Fixed: BRIN index for time-series performance (10x smaller than B-tree)
CREATE INDEX IF NOT EXISTS brin_flood_pixels_raw_first_seen 
ON flood_pixels_raw USING BRIN(first_seen);

CREATE INDEX IF NOT EXISTS idx_flood_pixels_raw_segment_id 
ON flood_pixels_raw(segment_id);

CREATE INDEX IF NOT EXISTS idx_flood_pixels_raw_geom 
ON flood_pixels_raw USING GIST(geom);

-- Deduplicated best-score per segment table
CREATE TABLE IF NOT EXISTS flood_pixels_unique (
    segment_id bigint PRIMARY KEY,
    score smallint NOT NULL,
    homes integer NOT NULL,
    qpe_1h numeric NOT NULL,
    ffw boolean NOT NULL,
    geom geometry(Point, 4326) NOT NULL,
    first_seen timestamptz NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- Create spatial index for unique table
CREATE INDEX IF NOT EXISTS idx_flood_pixels_unique_geom 
ON flood_pixels_unique USING GIST(geom);

-- Fixed: Partial index for marketable leads (future-proofing)
CREATE INDEX IF NOT EXISTS idx_flood_pixels_unique_marketable
ON flood_pixels_unique(score DESC)
WHERE score >= 40 AND homes >= 200;

-- Materialized view for marketable leads
CREATE MATERIALIZED VIEW IF NOT EXISTS flood_pixels_marketable AS
SELECT * FROM flood_pixels_unique
WHERE score >= 40 AND homes >= 200
ORDER BY score DESC 
LIMIT 10;

-- Create index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_flood_pixels_marketable_segment_id 
ON flood_pixels_marketable(segment_id);

-- Refresh function for materialized view
CREATE OR REPLACE FUNCTION refresh_flood_pixels_marketable()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY flood_pixels_marketable;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation (including segment_id encoding)
COMMENT ON TABLE flood_pixels_raw IS 'Append-only table for all flood pixel detections from Generator';
COMMENT ON TABLE flood_pixels_unique IS 'Deduplicated table maintaining highest score per segment within 3-hour window';
COMMENT ON MATERIALIZED VIEW flood_pixels_marketable IS 'Top 10 marketable flood leads (score ≥40, homes ≥200)';
COMMENT ON COLUMN flood_pixels_raw.segment_id IS 'Encoded as row*10000 + col from grid coordinates';
COMMENT ON COLUMN flood_pixels_unique.segment_id IS 'Encoded as row*10000 + col from grid coordinates';