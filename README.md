# Visualising Popular Songs using Self Organising Maps

Visualising the one million popular songs extracted from the Million Song
Dataset (the MillionSongSubset) [1]. The Self Organising Map is built using the
`minisom` library [2].

## How to use/build
1\. Get the SQLite [database](http://millionsongdataset.com/sites/default/files/AdditionalFiles/track_metadata.db) with the songs' metadata

2\. Export it to CSV filtering the invalid songs

```bash
sqlite3 -header -csv track_metadata.db < filter_data.sql > track_metadata.csv
```

The attribute `artist_mbid` is dropped from the dataset because it is only an
external identifier for the artist in the external *musicbrainz.org* database.

Songs without a year information are discarded.

515576 songs should be exported to the CSV

3\. Run the script


[1] BERTIN-MAHIEUX, Thierry. Million Song Dataset, official website.
Available at: [http://millionsongdataset.com/](http://millionsongdataset.com/).
Accessed on 05 May 2022.

[2] VETTIGLI, Giuseppe. MiniSom: minimalistic and NumPy-based implementation of
the Self Organizing Map. Available at:
[https://github.com/JustGlowing/minisom](https://github.com/JustGlowing/minisom).
Accessed on 05 May 2022.