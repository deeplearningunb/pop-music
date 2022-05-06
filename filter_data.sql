SELECT
	track_id, title, song_id, release, artist_id,
	artist_name, duration, artist_familiarity, artist_hotttnesss, year,
	track_7digitalid, shs_perf, shs_work
FROM
	songs
WHERE
	title IS NOT NULL AND title != ''
	AND release IS NOT NULL AND release != ''
	AND year IS NOT NULL AND year != 0
	LIMIT 10000;
