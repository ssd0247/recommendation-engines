# Why `data_collection` ?

* Dataset for training is downloaded (in compressed format) from open-source, freely available [Free Music Archive (FMA)][free-music-archive], by first signing on for an account & get the corresponding [API KEY][api-key], to access & download the audio files (legally licensed to be shared freely).

* Some pre-processing steps are needed to be performed for things like :
    * Building `artist`, `album` and `track` related folders.
    * Trimming the audio sample to only 30 secs etc.

---

# What does `create_dataset.py` do ?

1. Query the API and store metadata in `raw_tracks.csv`, `raw_albums.csv`, `raw_artists.csv` and `raw_genres.csv`.

2. Download the audio for each track.

3. Trim the audio to 30s clips.

4. Normalize the permissions and modification / access times.

5. Create the `.zip` archives.

---

<!--- Variables containing hyperlinks --->

[free-music-archive]: https://freemusicarchive.org

[api-key]: https://en.wikipedia.org/wiki/API_key

<!--- ******************************* --->