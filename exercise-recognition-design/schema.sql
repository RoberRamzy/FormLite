-- FormLite prototype: SQLite reference schema, version 1.
-- Implement equivalent Room entities/migrations; validate temporal invariants in repository.
PRAGMA foreign_keys = ON;

CREATE TABLE exercise (
    id TEXT PRIMARY KEY NOT NULL,
    display_name TEXT NOT NULL
);

CREATE TABLE session (
    id TEXT PRIMARY KEY NOT NULL,
    started_at_utc_ms INTEGER NOT NULL,
    ended_at_utc_ms INTEGER NOT NULL,
    active_duration_ms INTEGER NOT NULL CHECK (active_duration_ms >= 0)
);
-- UTC end need not exceed UTC start if the user changes the wall clock.
-- Active duration must always come from a monotonic timer.

CREATE TABLE recognition_segment (
    id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    exercise_id TEXT NOT NULL REFERENCES exercise(id) ON DELETE RESTRICT,
    start_active_ms INTEGER NOT NULL CHECK (start_active_ms >= 0),
    end_active_ms INTEGER NOT NULL CHECK (end_active_ms > start_active_ms)
);

CREATE INDEX idx_session_started_at ON session(started_at_utc_ms DESC);
CREATE INDEX idx_segment_session_start ON recognition_segment(session_id, start_active_ms);
CREATE INDEX idx_segment_exercise ON recognition_segment(exercise_id);

INSERT INTO exercise VALUES ('squat', 'Squats');
INSERT INTO exercise VALUES ('side_plank', 'Side plank');
INSERT INTO exercise VALUES ('reverse_crunches', 'Reverse crunches');
INSERT INTO exercise VALUES ('air_jump_rope', 'Air jump rope');

-- Application transaction must additionally enforce:
-- 1. All segment ends <= session.active_duration_ms.
-- 2. No overlapping segments in the same session.
-- 3. Save session + segments atomically using a stable session UUID.
-- 4. Total recognized time <= active duration.
-- 5. No raw video, pose frames, or model tensors stored in these tables.

CREATE VIEW session_exercise_duration AS
SELECT session_id, exercise_id,
       SUM(end_active_ms - start_active_ms) AS detected_duration_ms
FROM recognition_segment
GROUP BY session_id, exercise_id;

CREATE VIEW session_duration_summary AS
SELECT s.id AS session_id, s.active_duration_ms,
       COALESCE(SUM(r.end_active_ms - r.start_active_ms), 0) AS recognized_duration_ms,
       s.active_duration_ms - COALESCE(SUM(r.end_active_ms - r.start_active_ms), 0)
           AS unclassified_duration_ms
FROM session s
LEFT JOIN recognition_segment r ON r.session_id = s.id
GROUP BY s.id, s.active_duration_ms;
