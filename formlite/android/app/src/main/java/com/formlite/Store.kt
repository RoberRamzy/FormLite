package com.formlite

import android.content.Context
import androidx.room.ColumnInfo
import androidx.room.Dao
import androidx.room.Database
import androidx.room.DatabaseView
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.withTransaction

@Entity(tableName = "exercise")
internal data class ExerciseRow(@PrimaryKey val id: String, @ColumnInfo(name = "display_name") val name: String)

@Entity(tableName = "session", indices = [Index(
    value = ["started_at_utc_ms"], orders = [Index.Order.DESC], name = "idx_session_started_at",
)])
internal data class SessionRow(
    @PrimaryKey val id: String,
    @ColumnInfo(name = "started_at_utc_ms") val started: Long,
    @ColumnInfo(name = "ended_at_utc_ms") val ended: Long,
    @ColumnInfo(name = "active_duration_ms") val duration: Long,
)

@Entity(tableName = "recognition_segment", foreignKeys = [
    ForeignKey(entity = SessionRow::class, parentColumns = ["id"], childColumns = ["session_id"],
        onDelete = ForeignKey.CASCADE),
    ForeignKey(entity = ExerciseRow::class, parentColumns = ["id"], childColumns = ["exercise_id"],
        onDelete = ForeignKey.RESTRICT),
], indices = [
    Index(value = ["session_id", "start_active_ms"], name = "idx_segment_session_start"),
    Index(value = ["exercise_id"], name = "idx_segment_exercise"),
])
internal data class SegmentRow(
    @PrimaryKey val id: String,
    @ColumnInfo(name = "session_id") val sessionId: String,
    @ColumnInfo(name = "exercise_id") val exerciseId: String,
    @ColumnInfo(name = "start_active_ms") val start: Long,
    @ColumnInfo(name = "end_active_ms") val end: Long,
)

private const val EXERCISE_SUM = "SELECT session_id, exercise_id, SUM(end_active_ms - start_active_ms) AS detected_duration_ms FROM recognition_segment GROUP BY session_id, exercise_id"
private const val SESSION_SUM = "SELECT s.id AS session_id, s.active_duration_ms, COALESCE(SUM(r.end_active_ms - r.start_active_ms), 0) AS recognized_duration_ms, s.active_duration_ms - COALESCE(SUM(r.end_active_ms - r.start_active_ms), 0) AS unclassified_duration_ms FROM session s LEFT JOIN recognition_segment r ON r.session_id = s.id GROUP BY s.id, s.active_duration_ms"

@DatabaseView(value = EXERCISE_SUM, viewName = "session_exercise_duration")
internal data class ExerciseDuration(
    @ColumnInfo(name = "session_id") val sessionId: String,
    @ColumnInfo(name = "exercise_id") val exerciseId: String,
    @ColumnInfo(name = "detected_duration_ms") val duration: Long,
)

@DatabaseView(value = SESSION_SUM, viewName = "session_duration_summary")
internal data class SessionDuration(
    @ColumnInfo(name = "session_id") val sessionId: String,
    @ColumnInfo(name = "active_duration_ms") val duration: Long,
    @ColumnInfo(name = "recognized_duration_ms") val recognized: Long,
    @ColumnInfo(name = "unclassified_duration_ms") val unknown: Long,
)

@Dao
internal interface WorkoutDao {
    @Insert suspend fun insert(session: SessionRow)
    @Insert suspend fun insert(segments: List<SegmentRow>)
    @Query("SELECT * FROM session WHERE id = :id") suspend fun session(id: String): SessionRow?
    @Query("SELECT * FROM session ORDER BY started_at_utc_ms DESC, id") suspend fun sessions(): List<SessionRow>
    @Query("SELECT * FROM session_exercise_duration") suspend fun durations(): List<ExerciseDuration>
    @Query("SELECT * FROM session_exercise_duration WHERE session_id = :id")
    suspend fun durations(id: String): List<ExerciseDuration>
    @Query("SELECT * FROM session_duration_summary WHERE session_id = :id")
    suspend fun summary(id: String): SessionDuration?
    @Query("DELETE FROM session WHERE id = :id") suspend fun delete(id: String)
    @Query("DELETE FROM session") suspend fun deleteAll()
}

@Database(entities = [ExerciseRow::class, SessionRow::class, SegmentRow::class],
    views = [ExerciseDuration::class, SessionDuration::class], version = 1, exportSchema = true)
internal abstract class FormLiteDatabase : RoomDatabase() {
    abstract fun workouts(): WorkoutDao
}

internal class History(context: Context, databaseName: String = "formlite.db") {
    private val db = Room.databaseBuilder(context, FormLiteDatabase::class.java, databaseName)
        .createFromAsset("formlite.db").build()
    private val dao = db.workouts()

    suspend fun save(workout: Workout) = db.withTransaction {
        var priorEnd = 0L
        require(workout.duration >= 0)
        val sorted = workout.segments.sortedBy { it.start }
        for (segment in sorted) {
            require(segment.exerciseId in Motion.exerciseIds && segment.start >= priorEnd &&
                segment.end > segment.start && segment.end <= workout.duration)
            priorEnd = segment.end
        }
        if (dao.session(workout.id) == null) {
            dao.insert(SessionRow(workout.id, workout.started, workout.ended, workout.duration))
            dao.insert(sorted.map {
                SegmentRow(it.id, workout.id, it.exerciseId, it.start, it.end)
            })
        }
    }

    suspend fun list(): List<Pair<SessionRow, List<ExerciseDuration>>> = db.withTransaction {
        val durations = dao.durations().groupBy { it.sessionId }
        dao.sessions().map { it to durations[it.id].orEmpty() }
    }

    suspend fun detail(id: String): Triple<SessionRow, List<ExerciseDuration>, SessionDuration>? =
        db.withTransaction {
            val session = dao.session(id) ?: return@withTransaction null
            Triple(session, dao.durations(id), checkNotNull(dao.summary(id)))
        }

    suspend fun delete(id: String?) {
        if (id == null) dao.deleteAll() else dao.delete(id)
    }

    fun close() = db.close()
}
